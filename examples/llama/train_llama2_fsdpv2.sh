#!/bin/bash
###############################################################################
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################
# set -x

# set envs
export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_CHECKS_DISABLE=1
export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
export TOKENIZERS_PARALLELISM=false
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2

# parsing input arguments
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done


TIME_STAMP=$(date +"%Y-%m-%d_%H-%M-%S")
EXP_NAME="${EXP_NAME:-perf}"

NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens51np0}"
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ens51np0}"
TEE_OUTPUT="${TEE_OUTPUT:-1}"
NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-1}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-1}"
NO_TRAINING="${NO_TRAINING:-0}" # NO_TRAINING=1: for computing metrics only
ENABLE_PROFILING="${ENABLE_PROFILING:-0}"
echo "NO_TRAINING=$NO_TRAINING"

CWD=`pwd`
GPUS_PER_NODE=`python3 -c "import torch; print(torch.cuda.device_count())"`
# Change for multinode config
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-23731}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

MODEL_SIZE="${MODEL_SIZE:-70}"
TP="${TP:-1}"
PP="${PP:-1}"
CP="${CP:-1}"
MBS="${MBS:-1}"
BS="${BS:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
TOTAL_ITERS="${TOTAL_ITERS:-20}"
CONTI_PARAMS="${CONTI_PARAMS:-0}"
OPTIMIZER="${OPTIMIZER:-adam}"
TE_FP8="${TE_FP8:-0}"
GEMM_TUNING="${GEMM_TUNING:-1}"
MCORE="${MCORE:-1}"
FSDP="${FSDP:-1}"

EXPERIMENT_DIR="experiment"
mkdir -p $EXPERIMENT_DIR

DATA_DIR="${DATA_DIR:-/root/.cache/data}"
DATA_PATH="${DATA_PATH:-"$DATA_DIR/wikipedia_20220301.en/wikipedia_20220301.en.train.jsonl_text_document"}"

TOKENIZER_MODEL="${TOKENIZER_MODEL:-"$DATA_DIR/tokenizer_llama2"}"
# Download the tokenizer model
if ! [ -d "$TOKENIZER_MODEL" ]; then
  mkdir -p $TOKENIZER_MODEL
  wget -O $TOKENIZER_MODEL/special_tokens_map.json https://huggingface.co/NousResearch/Llama-2-7b-chat-hf/resolve/main/special_tokens_map.json
  wget -O $TOKENIZER_MODEL/tokenizer.json https://huggingface.co/NousResearch/Llama-2-7b-chat-hf/resolve/main/tokenizer.json
  wget -O $TOKENIZER_MODEL/tokenizer.model https://huggingface.co/NousResearch/Llama-2-7b-chat-hf/resolve/main/tokenizer.model
  wget -O $TOKENIZER_MODEL/tokenizer_config.json https://huggingface.co/NousResearch/Llama-2-7b-chat-hf/resolve/main/tokenizer_config.json

  echo "Tokenizer files downloaded successfully to $TOKENIZER_MODEL."
else
  echo "Folder $TOKENIZER_MODEL already exists. Skipping download."
fi

MAX_POSITION_EMBEDDINGS=128000

DEFAULT_LOG_DIR="${EXPERIMENT_DIR}/${NNODES}nodes_rank${NODE_RANK}_train_${MODEL_SIZE}B_mbs${MBS}_bs${BS}_tp${TP}_pp${PP}_cp${CP}_iter${TOTAL_ITERS}/nocompile${NO_TORCH_COMPILE}_TE_FP8_${TE_FP8}/${TIME_STAMP}"
LOG_DIR="${LOG_DIR:-${DEFAULT_LOG_DIR}}"
TRAIN_LOG="${LOG_DIR}/output_${EXP_NAME}.log"
mkdir -p $LOG_DIR
echo $TRAIN_LOG

if [ "$GEMM_TUNING" -eq 1 ]; then
    export TE_HIPBLASLT_TUNING_RUN_COUNT=20
    export TE_HIPBLASLT_TUNING_ALGO_COUNT=100
fi

if [ "$SEQ_LENGTH" -le 8192 ]; then
  ds_works=8
else
  ds_works=24
fi

if [[ $MODEL_SIZE -eq 8 ]]; then #llama3.1-8B
        HIDDEN_SIZE=4096 # e.g. llama-13b: 5120
        FFN_HIDDEN_SIZE=14336 # e.g. llama-13b: 13824
        NUM_LAYERS=32 # e.g. llama-13b: 40
        NUM_HEADS=32 # e.g. llama-13b: 40
        SEQ_LENGTH=$SEQ_LENGTH
        NUM_KV_HEADS=8 # llama3 70B uses GQA
elif [[ $MODEL_SIZE -eq 70 ]]; then
        HIDDEN_SIZE=8192 # e.g. llama-13b: 5120
        FFN_HIDDEN_SIZE=28672 # e.g. llama-13b: 13824
        NUM_LAYERS=80 # e.g. llama-13b: 40
        NUM_HEADS=64 # e.g. llama-13b: 40
        NUM_KV_HEADS=8 # llama3 70B uses GQA
        SEQ_LENGTH=$SEQ_LENGTH
        MAX_POSITION_EMBEDDINGS=$MAX_POSITION_EMBEDDINGS
else
        echo "Model size not supported."
        exit 1
fi

GROUP_SIZE=$(( ${NUM_HEADS} / ${NUM_KV_HEADS} ))
NUM_GROUPS=$(( ${NUM_HEADS} / ${GROUP_SIZE} ))

PROFILING_DIR="${LOG_DIR}/trace_${EXP_NAME}"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NUM_HEADS \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --no-position-embedding \
    --disable-bias-linear \
    --swiglu \
    --init-method-std 0.02 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --micro-batch-size $MBS \
    --global-batch-size $BS \
    --train-iters $TOTAL_ITERS \
    --no-async-tensor-model-parallel-allreduce \
    --bf16 \
    --no-masked-softmax-fusion \
    --disable-bias-linear \
    --recompute-num-layers 80 \
    --recompute-granularity full \
    --recompute-method block \
    --no-rope-fusion \
"
    
TRAIN_ARGS="
    --lr 1e-4 \
    --min-lr 1e-5 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --weight-decay 1.0e-1 \
    --clip-grad 1.0 \
    --ckpt-format torch_dist \
"


if [ "$OPTIMIZER" == "adam" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --optimizer adam \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        "
else
    TRAIN_ARGS="$TRAIN_ARGS --optimizer sgd \
        "
fi

DATA_ARGS="
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --split 99990,8,2 \
    --dataloader-type cyclic \
    --save-interval 200000 \
    --tensorboard-dir $LOG_DIR \
    --log-interval 1 \
    --eval-interval 320000 \
    --eval-iters 10 \
    --num-workers $ds_works \
    --data-path $DATA_PATH \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 5000 \
    --log-throughput \
    --no-save-optim \
    --eval-iters -1 \
"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"


CKPT_LOAD_ARGS="--exit-on-missing-checkpoint \
        --no-load-optim \
        --use-checkpoint-args \
        --no-load-rng"


EXTRA_ARGS="
    --group-query-attention \
    --num-query-groups $NUM_GROUPS \
    --no-gradient-accumulation-fusion \
    --distributed-backend nccl \
    --distributed-timeout-minutes 120 \
    --overlap-grad-reduce \
"


if [ "$FSDP" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --use-torch-fsdp2"
else
EXTRA_ARGS="$EXTRA_ARGS --use-distributed-optimizer --overlap-param-gather --sequence-parallel"
fi

if [ "$ENABLE_PROFILING" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --profile --use-pytorch-profiler --profile-step-start 10 --profile-step-end 12 --tensorboard-dir $LOG_DIR"
fi

if [ "$USE_FLASH_ATTN" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --use-flash-attn"
fi

if [ "$CONTI_PARAMS" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --use-contiguous-parameters-in-local-ddp"
fi

if [ "$MCORE" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --use-mcore-models"
fi

if [ "$TE_FP8" -eq 1 ]; then
EXTRA_ARGS="$EXTRA_ARGS --transformer-impl=transformer_engine \
    --fp8-margin=0 \
    --fp8-format=hybrid \
    --fp8-interval=1 \
    --fp8-amax-history-len=1024 \
    --fp8-amax-compute-algo=max \
    --attention-softmax-in-fp32 \
"
fi


run_cmd="
    torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        $EXTRA_ARGS \
        $TRAIN_ARGS \
"
echo '============='
echo $run_cmd
echo '============='

if [ "$TEE_OUTPUT" -eq 0 ]; then 
    run_cmd="$run_cmd >& $TRAIN_LOG"
else
    run_cmd="$run_cmd |& tee $TRAIN_LOG"
fi

if [ "$NO_TRAINING" -eq 0 ]; then 
    eval $run_cmd
fi


echo 'import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog="Process Log")
    parser.add_argument("filename")
    args = parser.parse_args()

    with open(args.filename) as f:
        lines = f.readlines()
    lines = lines[2:-1]
    lines = [float(a) for a in lines]
    mean = np.mean(np.array(lines))
    print(mean)' > mean_log_value.py


# echo '============================================================================================================'
grep -Eo 'throughput per GPU [^|]*' $TRAIN_LOG | sed -E 's/.*throughput per GPU \(TFLOP\/s\/GPU\): ([0-9\.]+).*/\1/' > tmp.txt
PERFORMANCE=$(python3 mean_log_value.py tmp.txt)
echo "throughput per GPU: $PERFORMANCE" |& tee -a $TRAIN_LOG
rm tmp.txt

# echo '============================================================================================================'
grep -Eo 'elapsed time per iteration [^|]*' $TRAIN_LOG | sed -E 's/.*elapsed time per iteration \(ms\): ([0-9\.]+).*/\1/' > tmp.txt
ETPI=$(python3 mean_log_value.py tmp.txt)
echo "elapsed time per iteration: $ETPI" |& tee -a $TRAIN_LOG

TIME_PER_ITER=$(python3 mean_log_value.py tmp.txt 2>/dev/null | awk '{printf "%.6f", $0}')
TGS=$(awk -v bs="$BS" -v sl="$SEQ_LENGTH" -v tpi="$TIME_PER_ITER" -v ws="$WORLD_SIZE" 'BEGIN {printf "%.6f", bs * sl * 1000/ (tpi * ws)}')
echo "tokens/GPU/s: $TGS" |& tee -a $TRAIN_LOG
rm tmp.txt

echo '============================================================================================================'
grep -Eo 'mem usages: [^|]*' $TRAIN_LOG | sed -E 's/.*mem usages: ([0-9\.]+).*/\1/' > tmp.txt
MEMUSAGE=$(python3 mean_log_value.py tmp.txt)
echo "mem usages: $MEMUSAGE" |& tee -a $TRAIN_LOG
rm tmp.txt



NUM_GROUPS=$(( ${NNODES} - 1 ))
if [[ $NODE_RANK -eq $NUM_GROUPS ]]; then
    'EXP_NAME	#Nodes	Model_SIZE 	Seq_Len	MBS	GBS	TP	PP	CP	Tokens/Sec/GPU	TFLOPs/s/GPU	Memory Usage	Time/iter'
    echo "${EXP_NAME}	$NNODES	$MODEL_SIZE	$SEQ_LENGTH	$MBS	$BS	$TP	$PP	$CP	$TGS	$PERFORMANCE	$MEMUSAGE	$ETPI" |& tee -a ../out.csv
    echo "${EXP_NAME}	$NNODES	$MODEL_SIZE	$SEQ_LENGTH	$MBS	$BS	$TP	$PP	$CP	$TGS	$PERFORMANCE	$MEMUSAGE	$ETPI" |& tee -a out.csv
else
        echo "Not the final node; check another the output for another node!"
        exit 1
fi