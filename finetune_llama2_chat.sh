TRAIN_DATA=/dockerx/wizard_platypus_sharegpt4/train_llama_s2048.jsonl
VALID_DATA=/dockerx/wizard_platypus_sharegpt4/train_llama_s2048.jsonl

TOKENIZER_MODEL=checkpoints/llama2_7b/hf
PRETRAINED_CHECKPOINT=checkpoints/llama2_7b/megatron
CHECKPOINT_PATH=checkpoints/llama2_7b/megatron_chat

GPUS_PER_NODE=`python -c "import torch; print(torch.cuda.device_count())"`
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6006
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

MODEL_SIZE="${MODEL_SIZE:-7}"
TP="${TP:-1}"
PP="${PP:-1}"
MBS="${MBS:-4}"
_BS=`python -c "import torch; print(int($GPUS_PER_NODE*$MBS))"`
BS="${BS:-$_BS}"
EPOCHS="${EPOCHS:-4}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
# total number of samples: 173658
_TOTAL_ITERS=`python -c "import math; print(int(math.floor(173658/$BS*$EPOCHS)))"`
TOTAL_ITERS="${TOTAL_ITERS:-$_TOTAL_ITERS}"

MAX_POSITION_EMBEDDINGS=4096

TRAIN_LOG="train_${MODEL_SIZE}B_iters${TOTAL_ITERS}_mbs${MBS}_bs${BS}_tp${TP}_pp${PP}_seq${SEQ_LENGTH}.log"

if [[ $MODEL_SIZE -eq 7 ]]; then
        HIDDEN_SIZE=4096 # e.g. llama-13b: 5120
        FFN_HIDDEN_SIZE=11008 # e.g. llama-13b: 13824
        NUM_LAYERS=32 # e.g. llama-13b: 40
        NUM_HEADS=32 # e.g. llama-13b: 40
        SEQ_LENGTH=$SEQ_LENGTH
        MAX_POSITION_EMBEDDINGS=$MAX_POSITION_EMBEDDINGS
        NUM_KV_HEADS=32 # llama2 70B uses GQA
elif [[ $MODEL_SIZE -eq 13 ]]; then
        HIDDEN_SIZE=5120 # e.g. llama-13b: 5120
        FFN_HIDDEN_SIZE=13824 # e.g. llama-13b: 13824
        NUM_LAYERS=40 # e.g. llama-13b: 40
        NUM_HEADS=40 # e.g. llama-13b: 40
        SEQ_LENGTH=$SEQ_LENGTH
        MAX_POSITION_EMBEDDINGS=$MAX_POSITION_EMBEDDINGS
        NUM_KV_HEADS=40 # llama2 70B uses GQA
elif [[ $MODEL_SIZE -eq 70 ]]; then
        HIDDEN_SIZE=8192 # e.g. llama-13b: 5120
        FFN_HIDDEN_SIZE=28672 # e.g. llama-13b: 13824
        NUM_LAYERS=80 # e.g. llama-13b: 40
        NUM_HEADS=64 # e.g. llama-13b: 40
        NUM_KV_HEADS=8 # llama2 70B uses GQA
        SEQ_LENGTH=$SEQ_LENGTH
        MAX_POSITION_EMBEDDINGS=$MAX_POSITION_EMBEDDINGS
else
        echo "Model size not supported."
        exit 1
fi

GROUP_SIZE=$(( ${NUM_HEADS} / ${NUM_KV_HEADS} ))
NUM_GROUPS=$(( ${NUM_HEADS} / ${GROUP_SIZE} ))


GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
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
    --bf16
"

TRAIN_ARGS="--lr 2.0e-5 \
        --lr-decay-iters 320000 \
        --lr-decay-style cosine \
        --weight-decay 1.0e-1 \
        --lr-warmup-fraction .001 \
	--adam-beta1 0.9 \
	--adam-beta2 0.95 \
        --clip-grad 1.0 \
        "

COMMON_TASK_ARGS_EXT="--train-data $TRAIN_DATA \
                      --valid-data $VALID_DATA \
                      --tokenizer-type HFTokenizer \
                      --tokenizer-model ${TOKENIZER_MODEL} \
                      --load $PRETRAINED_CHECKPOINT \
                      --dataloader-type cyclic \
                      --save-interval 1000 \
                      --tensorboard-dir $CHECKPOINT_PATH \
                      --save $CHECKPOINT_PATH \
                      --log-interval 100 \
                      --eval-interval 10000 \
                      --eval-iters 10"

CKPT_LOAD_ARGS="--exit-on-missing-checkpoint \
        --no-load-optim \
        --use-checkpoint-args \
        --no-load-rng"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

EXTRA_ARGS="
    --group-query-attention \
    --num-query-groups $NUM_GROUPS \
    --no-gradient-accumulation-fusion \
    --use-distributed-optimizer
"

torchrun $DISTRIBUTED_ARGS sft_llama2.py \
       --task GPT-CHAT \
       $CKPT_LOAD_ARGS \
       $GPT_ARGS \
       $TRAIN_ARGS \
       $COMMON_TASK_ARGS_EXT \
       $EXTRA_ARGS \
       $@

