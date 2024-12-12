# Pretrain sample for Qwen2 Model on ROCm Megatron-LM

## Table of Contents
   * [Dataset & Model Download](#dataset-and-model-download)
   * [Megatron-Core Model Training Process](#megatron-core-model-training-process)

## dataset-and-model-download

You can also use huggingfacecli to download the models in advance.
```bash

huggingface-cli download --token {your huggingface token} --resume-download Qwen/Qwen2-7B

mkdir qwen-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/wudao_qwenbpe_text_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/wudao_qwenbpe_text_document.idx

```

## megatron-core-model-training-process
```bash
MOCK_DATA=1 MODEL_SIZE=7B SEQ_LENGTH=1024 SEQ_PARALLEL=1 TP=1 PP=1 CP=1 GBS=8 MBS=1 RECOMPUTE_ACTIVATIONS=0 DIST_OPTIM=1 USE_FLASH_ATTN=1 TE_FP8=0 TOTAL_ITERS=3 bash ./pretrain_qwen2.sh
```