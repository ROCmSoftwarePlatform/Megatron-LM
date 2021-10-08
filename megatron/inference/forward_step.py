# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Forward step utilities."""

from collections.abc import Iterable

import torch

from megatron import (
    get_args,
    mpu)



class InferenceParams:


    def __init__(self, max_batch_size, max_sequence_len):

        self.max_sequence_len = max_sequence_len
        self.max_batch_size = max_batch_size
        self.sequence_len_offset = 0
        self.batch_size_offset = 0
        self.allocate_key_value_memory = True



class ForwardStep:

    def __init__(self, model, max_batch_size, max_sequence_len):

        # Make sure model is in eval mode.
        if isinstance(model, Iterable):
            for this_model in model:
                this_model.eval()
        else:
            model.eval()
        self.model = model

        self.constant = 512

        # Initialize inference parameters.
        self.inference_params = InferenceParams(max_batch_size,
                                                max_sequence_len)


    def __call__(self, tokens, position_ids, attention_mask):
        if tokens.size(0) * tokens.size(1) >= self.constant:
            micro_batch_size = max(1, self.constant // tokens.size(1))
            return _with_pipelining_forward_step(self.model, tokens,
                                                 position_ids,
                                                 attention_mask,
                                                 self.inference_params,
                                                 micro_batch_size)
        else:
            return _no_pipelining_forward_step(self.model, tokens,
                                               position_ids,
                                               attention_mask,
                                               self.inference_params)
            


def _get_recv_buffer_dtype(args):
    """Receive happens between the layers."""
    if args.fp32_residual_connection:
        return torch.float
    return args.params_dtype



def _allocate_recv_buffer(batch_size, sequence_length):
    """Receive happens between the layers with size [s, b, h]."""
    if mpu.is_pipeline_first_stage():
        return None
    args = get_args()
    recv_size = (sequence_length, batch_size, args.hidden_size)
    return torch.empty(recv_size,
                       dtype=_get_recv_buffer_dtype(args),
                       device=torch.cuda.current_device())



def _forward_step_helper(model, tokens, position_ids, attention_mask,
                         inference_params, recv_buffer=None):
    """Single forward step. Update the allocate memory flag so
    only the first time the memory is allocated."""
    batch_size = tokens.size(0)
    sequence_length = tokens.size(1)
    if recv_buffer is None:
        recv_buffer = _allocate_recv_buffer(batch_size, sequence_length)

    # Receive from previous stage.
    if not mpu.is_pipeline_first_stage():
        torch.distributed.recv(recv_buffer,
                               src=mpu.get_pipeline_model_parallel_prev_rank())

    # Forward pass through the model.
    model.set_input_tensor(recv_buffer)
    output_tensor = model(tokens, position_ids, attention_mask,
                          inference_params=inference_params)

    # Send output to the next stage.
    if not mpu.is_pipeline_last_stage():
        torch.distributed.send(output_tensor,
                               mpu.get_pipeline_model_parallel_next_rank())

    # Make sure we do not allocate context memory anymore.
    if inference_params.allocate_key_value_memory:
        inference_params.allocate_key_value_memory = False


    return output_tensor



def _no_pipelining_forward_step(model, tokens, position_ids, attention_mask,
                                inference_params, recv_buffer=None):

    # Run a simple forward pass.
    output_tensor = _forward_step_helper(model, tokens, position_ids,
                                         attention_mask, inference_params,
                                         recv_buffer=recv_buffer)
    # Update the sequence length offset.
    inference_params.sequence_len_offset += tokens.size(1)

    logits = None
    if mpu.is_pipeline_last_stage():
        logits = output_tensor

    return logits


def _with_pipelining_forward_step(model, tokens, position_ids, attention_mask,
                                  inference_params, micro_batch_size):

    sequence_length = tokens.size(1)
    batch_size = tokens.size(0)

    # Divide the batch dimension into micro batches.
    num_micro_batches, last_chunk = divmod(batch_size,
                                           micro_batch_size)
    if last_chunk > 0:
        num_micro_batches += 1

    # Preallocate memory for output logits.
    logits = None
    if mpu.is_pipeline_last_stage():
        args = get_args()
        logits = torch.empty(
            (batch_size, sequence_length, args.padded_vocab_size),
            dtype=torch.float32, device=torch.cuda.current_device())

    # Preallocate recv buffer.
    recv_buffer = _allocate_recv_buffer(micro_batch_size, sequence_length)

    for micro_batch_index in range(num_micro_batches):
        # Slice among the batch dimenion.
        start = micro_batch_index * micro_batch_size
        end = min(start + micro_batch_size, batch_size)
        this_micro_batch_size = end - start
        tokens2use = tokens[start:end, ...]
        position_ids2use = position_ids[start:end, ...]

        # Run a simple forward pass.
        if this_micro_batch_size != micro_batch_size:
            recv_buffer = None
        output = _forward_step_helper(model, tokens2use, position_ids2use,
                                      attention_mask, inference_params,
                                      recv_buffer=recv_buffer)

        # Adjust the batch size offset to account for the micro-batch.
        inference_params.batch_size_offset += this_micro_batch_size

        # Copy logits.
        if mpu.is_pipeline_last_stage():
            logits[start:end, ...] = output

    # Once we are done with all the micro-batches, we can
    # adjust the sequence length offset.
    inference_params.sequence_len_offset += sequence_length
    # and reset the batch size offset
    inference_params.batch_size_offset = 0

    return logits
