# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

""" Storage writer for PyT Distributed format allowing asynchronous save. """
import gc
import logging
import os
import queue
from contextlib import contextmanager
from itertools import chain
from pathlib import Path
from time import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import psutil
import torch
from torch import multiprocessing as mp
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.filesystem import DEFAULT_SUFFIX, _StoragePrefix, _write_item
from torch.distributed.checkpoint.planner import SavePlan, SavePlanner, WriteItem, WriteItemType
from torch.distributed.checkpoint.storage import WriteResult
from torch.futures import Future

logger = logging.getLogger(__name__)

WriteBucket = Tuple[Path, str, Tuple[list, list]]  # represents writes to a single file

_results_queue = None


def _get_write_results_queue():
    global _results_queue
    if _results_queue is None:
        ctx = mp.get_context('spawn')
        _results_queue = ctx.Manager().Queue()
    return _results_queue


@contextmanager
def _disable_gc():
    """Temporarily disables GC."""
    gc_enabled = gc.isenabled()
    try:
        if gc_enabled:
            gc.disable()
        yield
    finally:
        if gc_enabled:
            gc.enable()


class FileSystemWriterAsync(FileSystemWriter):
    """
    Async-enabled implementation of FileSystemWriter using file IO.

    This class doesn't spawn the async process itself, relies on the external async mechanism.

    Flow:
    1. Call `write_data`
    2. Externally start async process with `get_save_function_and_args` function and args
    3. The async function to call is `writer_proxy_func` which calls
       `write_preloaded_data` in multiple processes

    After saving is finalized on all ranks:
    4. Call `super().finish` with the results gathered in `self.writer_result`

    Note that step (3) above can also be called synchronously.

    Currently, it's assumed that a separate writer is created for each ckpt save
    (intermediate state is stored as writer attributes).
    """

    def __init__(self, *args, separation_hint: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.single_file_per_rank:
            raise NotImplementedError(
                'single_file_per_rank flag not supported for FileSystemWriterAsync'
            )

        # Intermediate state between preparation and finalization
        self.write_buckets: Optional[List[WriteBucket]] = None
        self.results_queue: Optional[mp.Queue] = None
        self.separation_hint = separation_hint

    def prepare_write_data(self, plan: SavePlan, planner: SavePlanner) -> None:
        """
        First stage of async saving. Copy data to CPU and plan the local saving.

        Args:
            plan (SavePlan): save plan generated by the PyT Distributed compatible planner
            planner (SavePlanner): save planner used to resolve the bytes and tensor data

        Returns: None, but stores the save plan in `self.write_buckets`
        """
        storage_plan: _StoragePrefix = plan.storage_data
        start = time()
        logger.debug(f"thread_count: {self.thread_count}, time: {start}")
        if self.separation_hint:
            assert (
                self.thread_count > 1
            ), "thread_count must be at least 2 if separation_hint is provided"
        bins = self.thread_count // 2 if self.separation_hint is not None else self.thread_count
        item_buckets = _split_by_size_and_type(bins, plan.items, self.separation_hint)
        logger.debug(f"bucket_prep, time: {time() - start}")

        start = time()
        # move tensors from GPU to CPU before starting async writing
        # We do D2H synchronously for now
        file_count = 0

        def gen_file(prefix=""):
            nonlocal file_count
            file_name = f"{prefix}{storage_plan.prefix}{file_count}{DEFAULT_SUFFIX}"
            file_count += 1
            return file_name

        # Prepare bytes / tensor data in each bucket, which will be assigned to each writer process
        self.write_buckets = []
        for group_name, group_buckets in _split_by_separation_hint(
            item_buckets, self.separation_hint
        ).items():
            for bucket in group_buckets:
                bytes_data = [
                    (item, planner.resolve_data(item))
                    for item in bucket
                    if item.type == WriteItemType.BYTE_IO
                ]
                tensor_data = [
                    (item, planner.resolve_data(item).detach().to("cpu", non_blocking=True))
                    for item in bucket
                    if item.type != WriteItemType.BYTE_IO
                ]
                if len(bytes_data) > 0 or len(tensor_data) > 0:
                    file_name = gen_file(prefix=group_name)
                    self.write_buckets.append(
                        (self.path / file_name, file_name, (bytes_data, tensor_data))
                    )

        # Check if there is anything to write on this rank
        if len(self.write_buckets) > 0:
            assert len(self.write_buckets) <= self.thread_count, (
                len(self.write_buckets),
                self.thread_count,
            )
            self.results_queue = _get_write_results_queue()
        else:
            self.results_queue = None
        end = time()
        logger.debug(f"D2H and push, time: {end - start}")

    def get_save_function_and_args(self) -> Tuple[Optional[Callable], Tuple]:
        """
        Get function that saves the data to storage along with its arguments.
        Allows the external caller to apply the save function synchronously or asynchronously.

        Returns: None (if there is nothing to write on this rank) or a tuple of:
            - the function that saves the data
            - arguments to that function
        """
        if not self.write_buckets:
            return None, ()
        return (self.write_preloaded_data_multiproc, (self.write_buckets, self.results_queue))

    @staticmethod
    @_disable_gc()
    def write_preloaded_data_multiproc(
        write_buckets: List[WriteBucket], global_results_queue: mp.Queue
    ) -> None:
        """
        Performs saving data to storage with multiple processes.

        Starts predefined number of processes and uses 2 queues to make sure the results
        are complete:
        - local_results_queue - to send the actual results
        - count_queue - small queue to mark worker as completed

        Using just one queue disallowed proper exception handling.

        This method is meant to be run in a forked subprocess.
        Triggering GC during execution leads to CUDA errors
        (cleaning up tensors owned by the parent process).
        To prevent this, we disable the GC explicitly for this function with _disable_gc.

        Args:
            write_buckets (List[WriteBucket]): write plan
            global_results_queue (mp.Queue): mp.Queue to collect Dict[List[WriteResults]]
                (or an Exception) from parallel write processes to the main training process
        Returns: None
        """
        w_start = time()
        write_results_or_exc: Union[dict, Exception] = dict()
        ctx = mp.get_context('spawn')
        torch.distributed.init_process_group()
        local_results_queue = ctx.Queue()
        count_queue = ctx.JoinableQueue()
        p_list = []
        for i, write_bucket in enumerate(write_buckets):
            try:
                count_queue.put(i)
                p_list.append(
                    ctx.Process(
                        target=FileSystemWriterAsync.write_preloaded_data,
                        args=(i, write_bucket, local_results_queue, count_queue, True),
                    )
                )
            except Exception as e:
                err_msg = f'An error is caught while a proc {i} is created, error: {e}'
                logger.error(err_msg)
                write_results_or_exc = RuntimeError(err_msg)

        if not isinstance(write_results_or_exc, Exception):
            for p in p_list:
                p.start()

            logger.debug('FileSystemWriterAsync: collecting worker results...')

            # To make sure all nodes are completed
            count_queue.join()
            # At this point, all workers completed, so the queue should have exactly
            # `len(write_buckets)` items
            for proc_idx in range(len(write_buckets)):
                try:
                    local_proc_idx, local_results_or_exc = local_results_queue.get()
                except queue.Empty:
                    write_results_or_exc = RuntimeError(
                        f'Unexpected empty `local_results_queue`'
                        f' (got only {proc_idx}/{len(write_buckets)} items)'
                    )
                    break
                else:
                    if isinstance(local_results_or_exc, Exception):
                        err_msg = (
                            f"Local process {local_proc_idx} encountered"
                            f" an error: {local_results_or_exc}"
                        )
                        logger.error(err_msg)
                        write_results_or_exc = local_results_or_exc
                        break
                    else:
                        assert isinstance(local_results_or_exc, list), type(local_results_or_exc)
                        write_results_or_exc[local_proc_idx] = local_results_or_exc
                        p_list[local_proc_idx].join()

            logger.debug('FileSystemWriterAsync: collected worker results successfully')

        global_results_queue.put(write_results_or_exc)

        w_end = time()
        logger.debug(
            f"{w_end}, rank: {torch.distributed.get_rank()},"
            f" write(sync,parallel): {w_end - w_start}"
        )

    @staticmethod
    @_disable_gc()
    def write_preloaded_data(
        local_proc_idx: int,
        write_bucket: WriteBucket,
        results_queue: mp.SimpleQueue,
        count_queue: mp.JoinableQueue,
        use_fsync: bool,
    ) -> None:
        """
        Performs actual data saving to storage.

        Args:
            local_proc_idx (int): index of a local process that performs writing
            write_bucket (WriteBucket): data to write to storage
            results_queue (mp.Queue): queue to return the write results
                to the proxy checkpoint process.
            count_queue (mp.JoinableQueue): queue to marks worker task as completed
            use_fsync (bool): if True, calls os.fsync at the end of saving

        Returns: None, the write result are put into the `queue`
        """
        mem_before = _process_memory()

        local_results = []
        try:
            file_name, storage_key, (bytes_data, tensor_data) = write_bucket
            with open(file_name, "wb") as stream:
                for write_item, data in bytes_data:
                    local_results.append(_write_item(stream, data, write_item, storage_key))

                for write_item, tensor in tensor_data:
                    assert tensor.is_cpu
                    local_results.append(_write_item(stream, tensor, write_item, storage_key))

                if use_fsync:
                    os.fsync(stream.fileno())
            local_output = (local_proc_idx, local_results)
        except Exception as e:
            local_output = (local_proc_idx, e)

        results_queue.put(local_output)
        # Signal this process is done.
        count_queue.get()
        count_queue.task_done()

        mem_after = _process_memory()
        logger.debug(
            f"{local_proc_idx} consumed: {mem_after - mem_before},"
            f" before: {mem_before}, after: {mem_after}"
        )

    def write_data(self, plan: SavePlan, planner: SavePlanner) -> Future[List[WriteResult]]:
        """Write all items from ``plan``."""
        raise NotImplementedError('write_data not implemented for FileSystemWriterAsync')

    def retrieve_write_results(self) -> List[WriteResult]:
        """
        Turn the latest dict including write results from `self.results_queue`
            into a single results lists. Includes error check.

        Returns (List[WriteResult]): the list of write results
            from all local processes performing the save.

        """
        assert self.write_buckets is not None

        if self.results_queue is None:
            write_results_or_exc = {}
        else:
            try:
                write_results_or_exc = self.results_queue.get_nowait()
            except queue.Empty:
                raise RuntimeError(f'results_queue should not be empty')

        if isinstance(write_results_or_exc, Exception):
            raise RuntimeError(f'Worker failure: {write_results_or_exc}') from write_results_or_exc
        write_results: dict = write_results_or_exc
        if len(write_results) != len(self.write_buckets):
            raise RuntimeError(
                f'Incomplete worker results (expected {len(self.write_buckets)},'
                f' got {len(write_results)}. This probably indicates a worker failure.'
            )
        return list(chain.from_iterable(write_results.values()))


def _split_by_size_and_type(
    bins: int, items: List[WriteItem], separation_hint: Optional[str] = None
) -> List[List[WriteItem]]:
    """
    Splits write items according to item size into close to uniform bins.

    Same as torch.distributed.checkpoint.filesystem._split_by_size_and_type,
    but with a fixed _item_size function.

    Args:
        bins (int): numbers of bins to split to
        items (List[WriteItem]): list of write items

    Returns (List[List[WriteItem]]): write items split to bins
    """
    if bins == 1:
        return [items]

    bytes_items = [wi for wi in items if wi.type == WriteItemType.BYTE_IO]
    tensor_items = [wi for wi in items if wi.type != WriteItemType.BYTE_IO]

    buckets: List[List[WriteItem]] = [[] for _ in range(bins)]
    bucket_sizes = [0 for _ in range(bins)]

    tensor_items.sort(key=_item_size, reverse=True)

    # Assign bytes with a simple round-robin
    for i, item in enumerate(bytes_items):
        buckets[i % bins].append(item)

    # Then, assign tensors according to their sizes
    for item in tensor_items:
        # TODO replace with headq
        idx = min(enumerate(bucket_sizes), key=lambda x: x[1])[0]
        buckets[idx].append(item)
        bucket_sizes[idx] += _item_size(item)

    return buckets


def _split_by_separation_hint(
    buckets: List[List[WriteItem]], separation_hint: Optional[str] = None
) -> Dict[str, List[List[WriteItem]]]:
    """
    Splits buckets into those whose keys begin with the separation_hint and those whose keys do not

    Args:
        buckets (List[List[WriteItem]]): buckets to split
        separation_hint (Optional[str]): optional prefix to split on

    Returns (Dict[str, List[List[WriteItem]]]): a dictionary
        mapping the prefix to the relevant buckets
    """
    bins = len(buckets)
    buckets_with_separation_hint = {}
    if separation_hint is not None:
        buckets_default = [[] for _ in range(bins)]
        buckets_hint = [[] for _ in range(bins)]
        for i in range(bins):
            for item in buckets[i]:
                if item.index.fqn.startswith(separation_hint):
                    buckets_hint[i].append(item)
                else:
                    buckets_default[i].append(item)
        buckets_with_separation_hint[""] = buckets_default
        buckets_with_separation_hint[separation_hint] = buckets_hint
    else:
        buckets_with_separation_hint[""] = buckets
    return buckets_with_separation_hint


def _item_size(item: WriteItem) -> int:
    """
    Calculates size (in bytes) of a single write item.

    Same as torch.distributed.checkpoint.filesystem._item_size,
    but fixes computing chunk size (with item.tensor_data.chunk.sizes)

    Args:
        item (WriteItem): write item to compute the size of

    Returns (int): size of an item in bytes
    """
    size = 1
    assert item.tensor_data is not None
    # can't use math.prod as PT needs to support older python
    for s in item.tensor_data.chunk.sizes:
        size *= s

    dtype = item.tensor_data.properties.dtype
    return size * torch._utils._element_size(dtype)


def _process_memory() -> int:
    """
    Get memory used by current process.

    Returns (int): memory used by current process
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss
