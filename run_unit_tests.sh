#!/bin/bash

set -x
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 -m pytest -m "not flaky and not nternal and not failing_on_rocm_mi250 and not failing_on_rocm" --csv test_report.csv Megatron-LM/tests/unit_tests/