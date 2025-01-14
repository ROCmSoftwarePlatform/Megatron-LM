#!/bin/bash

# Based on https://github.com/AMD-AIG-AIMA/AMD-Megatron-LM/blob/core0.8.0_guihong/train_acc_loss_llama3.sh

# set -x

export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_CHECKS_DISABLE=1
export NCCL_IB_HCA=mlx5_ib
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
# The 2 lines below should be commented if running on AAC.
#export NCCL_SOCKET_IFNAME=ib0
#export GLOO_SOCKET_IFNAME=ib0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
export TOKENIZERS_PARALLELISM=false
# export AMD_LOG_LEVEL=3
# export AMD_SERIALIZE_KERNEL=3
export HSA_NO_SCRATCH_RECLAIM=1


# export RCCL_MSCCLPP_ENABLE=0
# export HSA_ENABLE_IPC_MODE_LEGACY=1

#export NCCL_MIN_NCHANNELS=112


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

TEE_OUTPUT="${TEE_OUTPUT:-1}"
NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-1}"
NO_TRAINING="${NO_TRAINING:-0}" # NO_TRAINING=1: for computing metrics only
ENABLE_PROFILING="${ENABLE_PROFILING:-0}"
echo "NO_TRAINING=$NO_TRAINING"

GPUS_PER_NODE=`python3 -c "import torch; print(torch.cuda.device_count())"`
# GPUS_PER_NODE=1

# Change for multinode config
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-23733}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

MODEL_SIZE="${MODEL_SIZE:-8}"
TP="${TP:-1}"
PP="${PP:-1}"
PP_VP="${PP_VP:-None}"
CP="${CP:-1}"
MBS="${MBS:-1}"
GBS="${GBS:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-131072}"
TOTAL_ITERS="${TOTAL_ITERS:-5}"
SEQ_PARALLEL="${SEQ_PARALLEL:-1}" 
CONTI_PARAMS="${CONTI_PARAMS:-0}"
OPTIMIZER="${OPTIMIZER:-adam}"
TE_FP8="${TE_FP8:-0}"
GEMM_TUNING="${GEMM_TUNING:-1}"
MOCK_DATA="${MOCK_DATA:-0}"
AC=${AC:-sel}
DO=${DO:-true}
FL=${FL:-true}
TE=${TE:-true}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-32}

EXPERIMENT_DIR="experiment"
mkdir -p $EXPERIMENT_DIR

DATA_PATH=../../../fineweb-edu/fineweb-edu-train_text_document

if [ $MODEL_SIZE = 8 ]; then
TOKENIZER_MODEL=examples/llama3/tokenizers/Llama-3.1-8B
elif [ $MODEL_SIZE = 70 ]; then
TOKENIZER_MODEL=examples/llama3/tokenizers/Llama-3.1-70B
elif [ $MODEL_SIZE = 405 ]; then
TOKENIZER_MODEL=examples/llama3/tokenizers/Llama-3.1-405B
fi

DEFAULT_LOG_DIR="${EXPERIMENT_DIR}/${NNODES}nodes_rank${NODE_RANK}_train_${MODEL_SIZE}B_mbs${MBS}_gbs${GBS}_seqlen${SEQ_LENGTH}_tp${TP}_pp${PP}_cp${CP}_iter${TOTAL_ITERS}_SL_${SEQ_PARALLEL}_AC_${AC}_DO_${DO}_FL_${FL}_TE_${TE}/nocompile${NO_TORCH_COMPILE}_TE_FP8_${TE_FP8}/${TIME_STAMP}"
LOG_DIR="${LOG_DIR:-${DEFAULT_LOG_DIR}}"
TRAIN_LOG="${LOG_DIR}/output_${EXP_NAME}.log"
echo "Writing to LOG_DIR: ${LOG_DIR} ..."

mkdir -p $LOG_DIR
echo $TRAIN_LOG

if [ "$GEMM_TUNING" -eq 1 ]; then
    export TE_HIPBLASLT_TUNING_RUN_COUNT=10
    export TE_HIPBLASLT_TUNING_ALGO_COUNT=50
fi

if [ "$SEQ_LENGTH" -le 8192 ]; then
    ds_works=8
else
    ds_works=24
fi

if [ $MODEL_SIZE = 8 ]; then

HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=14336
NUM_LAYERS=32
NUM_HEADS=32
NUM_KV_HEADS=8

elif [ $MODEL_SIZE = 70 ]; then

HIDDEN_SIZE=8192
FFN_HIDDEN_SIZE=28672
NUM_LAYERS=80
NUM_HEADS=64
NUM_KV_HEADS=8

elif [ $MODEL_SIZE = 405 ]; then

HIDDEN_SIZE=16384 
FFN_HIDDEN_SIZE=53248
NUM_LAYERS=16 # e.g. llama-405B 126 layers, for 16 layers, 50B model size
NUM_HEADS=128
NUM_KV_HEADS=8
MAX_POSITION_EMBEDDINGS=131072

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
    --global-batch-size $GBS \
    --train-iters $TOTAL_ITERS \
    --no-async-tensor-model-parallel-allreduce \
    --bf16 \
    --no-masked-softmax-fusion \
    --disable-bias-linear \
    --no-create-attention-mask-in-dataloader \
"

if [ "$PP_VP" != "None" ]; then
    GPT_ARGS="$GPT_ARGS --num-layers-per-virtual-pipeline-stage ${PP_VP}"
fi

TRAIN_ARGS="--lr 1e-4 \
        --min-lr 1e-5 \
        --lr-decay-iters 320000 \
        --lr-decay-style cosine \
        --weight-decay 1.0e-1 \
        --clip-grad 1.0 \
        --optimizer ${OPTIMIZER}
"
        # --lr-warmup-fraction .001 \
        # --adam-beta1 0.9 \
        # --adam-beta2 0.95 \

DATA_ARGS="
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --dataloader-type cyclic \
    --num-workers $ds_works \
"

if [ "$MOCK_DATA" -eq 1 ]; then
    echo Using mock data.
    DATA_ARGS="$DATA_ARGS --mock-data"
else
    echo Using data from $DATA_PATH
    DATA_ARGS="$DATA_ARGS --data-path $DATA_PATH"
fi

OUTPUT_ARGS="
    --log-interval 1 \
    --log-throughput \
    --no-save-optim \
    --eval-iters -1 \
    --tensorboard-dir $LOG_DIR
"
#    --save-interval 200000 \
#    --eval-interval 320000 \
#    --eval-iters 10 \
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
    --distributed-timeout-minutes 120 \
    --overlap-grad-reduce \
"

if [ $AC = full ]; then
    EXTRA_ARGS="$EXTRA_ARGS --recompute-method uniform --recompute-granularity full --recompute-num-layers ${RECOMPUTE_NUM_LAYERS}"
elif [ $AC = sel ]; then
    EXTRA_ARGS="$EXTRA_ARGS --recompute-activations"
fi

if [ $DO = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --use-distributed-optimizer --overlap-param-gather"
fi

if [ $FL = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --use-flash-attn"
fi

if [ $TE = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --transformer-impl transformer_engine"
elif [ $TE = false ]; then
    EXTRA_ARGS="$EXTRA_ARGS --transformer-impl local"
fi


if [ "$ENABLE_PROFILING" -eq 1 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --profile --use-pytorch-profiler"
fi

if [ "$SEQ_PARALLEL" -eq 1 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --sequence-parallel"
fi

if [ "$CONTI_PARAMS" -eq 1 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --use-contiguous-parameters-in-local-ddp"
fi

if [ "$TE_FP8" -eq 1 ]; then
    EXTRA_ARGS="$EXTRA_ARGS \
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

run_cmd="$run_cmd 2>&1 | tee $TRAIN_LOG"

eval $run_cmd

echo '============================================================================================================' |& tee -a $TRAIN_LOG
PERFORMANCE=$(grep -Eo 'throughput per GPU [^|]*' $TRAIN_LOG | sed -E 's/.*throughput per GPU \(TFLOP\/s\/GPU\): ([0-9\.]+).*/\1/' |  awk 'NR > 2 { sum += $1; count++ } END { if (count > 0) printf sum / count }')
echo "throughput per GPU: $PERFORMANCE" |& tee -a $TRAIN_LOG

ETPI=$(grep -Eo 'elapsed time per iteration [^|]*' $TRAIN_LOG | sed -E 's/.*elapsed time per iteration \(ms\): ([0-9\.]+).*/\1/' |  awk 'NR > 2 { sum += $1; count++ } END { if (count > 0) print sum / count }')
echo "elapsed time per iteration: $ETPI" |& tee -a $TRAIN_LOG

TGS=$(awk -v gbs="$GBS" -v sl="$SEQ_LENGTH" -v tpi="$ETPI" -v ws="$WORLD_SIZE" 'BEGIN {printf "%.6f", gbs * sl * 1000/ (tpi * ws)}')
echo "tokens/GPU/s: $TGS" |& tee -a $TRAIN_LOG
echo '============================================================================================================' |& tee -a $TRAIN_LOG
