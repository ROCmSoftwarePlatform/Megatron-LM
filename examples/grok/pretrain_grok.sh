#!/bin/bash

# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

export CUDA_DEVICE_MAX_CONNECTIONS=1

export NCCL_CHECKS_DISABLE=1

export NVTE_CK_V3_ATOMIC_FP32=0
export NVTE_CK_V3_BF16_CVT=1
export NVTE_CK_V3_SPEC=1
export NVTE_CK_USES_BWD_V3=1

export TE_HIPBLASLT_TUNING_ALGO_COUNT=50
export TE_HIPBLASLT_TUNING_RUN_COUNT=10

export TORCH_NCCL_HIGH_PRIORITY=1

# Script arguments
TOKENIZER=$1
DATA_PATH=$2

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NNODES=${SLURM_NNODES:-1}
NODE_RANK=${RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

TP=${TP:-1}
PP=${PP:-1}
CP=${CP:-1}
EP=${EP:-1}
MBS=${MBS:-1}
BS=${BS:-$(($GPUS_PER_NODE*$MBS))}
TOTAL_ITERS=${TOTAL_ITERS:-20}

LOG_PATH=logs/train/`uname -n`_`date +%Y%m%d-%H%M`
SAVE_PATH=$LOG_PATH/checkpoints
mkdir -p $SAVE_PATH

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    # Embeddings
    --no-position-embedding
    --position-embedding-type rope
    --rotary-base 10000

    # Normalization
    --normalization RMSNorm

    # Attention
    --attention-dropout 0
    --group-query-attention
    --num-attention-heads 48
    --num-query-groups 8

    # MoE
    --ffn-hidden-size 32768
    --moe-aux-loss-coeff 1e-3
    --moe-z-loss-coeff 1e-3
    --num-experts 8
    --hidden-dropout 0
    --hidden-size 6144
    --moe-token-dispatcher-type alltoall
    --disable-bias-linear
    --moe-pad-expert-input-to-capacity
    --moe-expert-capacity-factor 1.25

    # Global
    --num-layers 64
)

DATA_ARGS=(
    --max-position-embeddings 8192
    --seq-length 8192
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER
    --data-path $DATA_PATH
    --num-workers 24
)

TRAINING_ARGS=(
    --micro-batch-size $MBS
    --global-batch-size $BS
    --lr 1e-4
    --train-iters $TOTAL_ITERS
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --clip-grad 1.0
    --no-gradient-accumulation-fusion
    --bf16
    --fp8-margin 0
    --fp8-format hybrid
    --fp8-interval 1
    --fp8-amax-history-len 1024
    --fp8-amax-compute-algo max
    --attention-softmax-in-fp32
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
    --context-parallel-size $CP
    --expert-model-parallel-size $EP
    --sequence-parallel
    --use-distributed-optimizer
    --overlap-param-gather
    --overlap-grad-reduce
)

LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --eval-iters -1
    --no-load-optim
    --no-load-rng
    --ckpt-format torch
    --save $SAVE_PATH
    --save-interval 500
    --tensorboard-dir $LOG_PATH
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_grok.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    $@ |& tee $LOG_PATH/output.log
