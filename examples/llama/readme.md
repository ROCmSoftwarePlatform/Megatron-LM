# Llama2/Llama3 Model Pretraining Instructions

This guide provides the steps for setting up the environment and configuring the script to train Llama2 or Llama3 models.

---

## 1. Environment Setup

1. **Download Docker Image**  
   Download the Docker image required for training:  
   `docker pull <image_name>`

2. **Launch Docker Container**  
   Start the Docker container:  
   `docker run -it --device /dev/dri --device /dev/kfd --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v $HOME:$HOME -v  $HOME/.ssh:/root/.ssh --shm-size 64G --name megatron_training_env <image_name>`

3. **Prepare training datasets**
   If you already have the preprocessed data, you can skip this section.
   
   Use the following command to process datasets. We use GPT data as an example. You may change the merge table, use an end-of-document token, remove sentence splitting, and use the tokenizer type.

  ```bash
  python tools/preprocess_data.py \
    --input my-corpus.json \
    --output-prefix my-gpt2 \
    --vocab-file gpt2-vocab.json \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file gpt2-merges.txt \
    --append-eod
  ```
  In this case, the automatically generated output files are named `my-gpt2_text_document.bin` and `my-gpt2_text_document.idx`.

---

## 2. Configurations in Script (`Megatron-LM/examples/llama`)
Use `train_llama3.sh` for Llama3/3.1 models and `train_llama2.sh` for Llama2 models.

### 2.1 Network Interface
Update the network interface in the script to match your system’s network interface.
To find your network interface, run (out of container):
```bash
ip a
```
Then, update the following variables in the script:
```bash
export NCCL_SOCKET_IFNAME=ens50f0np0
export GLOO_SOCKET_IFNAME=ens50f0np0
```

### 2.2 Dataset
You can use either mock data or real data for training.

- **Mock Data:**
  Use `MOCK_DATA` variable to toggle between mock and real data. Default value is 1. 
  ```bash
  MOCK_DATA=1 
  ```
- **Real Data:**
  Update the `DATA_PATH` to the location where your dataset is stored:
  ```bash
  MOCK_DATA=0
  DATA_DIR="/root/.cache/data"  # Change to where your dataset is stored
  DATA_PATH=${DATA_DIR}/bookcorpus_text_sentence
  ```

### 2.3 Tokenizer

- **For Llama2 Training:**
  Use the `Llama2Tokenizer`.

- **For Llama3 Training:**
  Use the `HuggingFaceTokenizer`. Set the HuggingFace model path in the `TOKENIZER_MODEL` variable:
  ```bash
  TOKENIZER_MODEL=meta-llama/Llama-3.1-8B  # For Llama3
  ```

### 2.4 Multi-node Training
If you're running multi-node training, update the following environment variables on each node.They can also be passed as command line arguments.

- **Master Address:**
  Change `localhost` to the master node's hostname:
  ```bash
  MASTER_ADDR="${MASTER_ADDR:-localhost}"
  ```

- **Number of Nodes:**
  Set the number of nodes you want to train on (e.g., 2, 4, 8):
  ```bash
  NNODES="${NNODES:-1}"
  ```

- **Node Rank:**
  Set the rank of each node (0 for master, 1 for the first worker node, etc.):
  ```bash
  NODE_RANK="${NODE_RANK:-0}"
  ```

- **DATA_CACHE_PATH:**
  Set `DATA_CACHE_PATH` to a common directory accessible by all the nodes (for eg, an NFS directory) for multi-node runs
  ```bash
  DATA_CACHE_PATH=/root/cache #Set to a common directory for multi-node runs
  ```

 - **Network Drivers Inside Docker:** 
   For multi-node runs, make sure correct network drivers are installed on the nodes. If inside a docker, either install the drivers inside the docker container or pass the network drivers from the host while creating docker container.


## 3. How to Run

### 3.1 Single Node Training
To run the training on a single node, go to Megatron-LM folder, use the following command:
```bash
TEE_OUTPUT=1 MBS=2 BS=128 TP=1 TE_FP8=1 SEQ_LENGTH=8192 MODEL_SIZE=8  bash examples/llama/train_llama3.sh
```

### 3.2 Multi-node Training
To run training on multiple nodes, launch the Docker container on each node. Example, follow these steps for 2 Node run with Node0 as master node :

- **On the Master Node0:**
  ```bash
  TEE_OUTPUT=1 MBS=2 BS=256 TP=1 TE_FP8=1 SEQ_LENGTH=8192 MODEL_SIZE=8  MASTER_ADDR=IP_NODE0 NNODES=2 NODE_RANK=0 bash examples/llama/train_llama3.sh
  ```

- **On the Worker Node1:**
  ```bash
  TEE_OUTPUT=1 MBS=2 BS=256 TP=1 TE_FP8=1 SEQ_LENGTH=8192 MODEL_SIZE=8  MASTER_ADDR=IP_NODE0 NNODES=2 NODE_RANK=1 bash examples/llama/train_llama3.sh
  ```
---

## 4. Key Variables to Pay Attention To

- **TE_FP8:**  
  `0` for BP16 (default), `1` for FP8.

- **GEMM_TUNING:**  
  `1` to enable GEMM tuning, which boosts performance by using the best GEMM kernels.

- **USE_FLASH_ATTN:**  
  `1` to enable Flash Attention.

- **ENABLE_PROFILING:**  
  `1` to enable PyTorch profiling for performance analysis.

- **transformer-impl:**  
  `transformer_engine` to use the Transformer Engine (TE). Set to `local` if you want to disable TE.

- **MODEL_SIZE:**  
  Set to `7B` or `70B` for Llama2, or `8B` or `70B` for Llama3/3.1.

- **TOTAL_ITERS:**  
  Set the total number of iterations (default: 10).

- **MOCK_DATA:**
  Use MOCK_DATA if set to 1, otherwise use the real data provided by user (DEFAULT: 1)

- **MBS:**
  Micro batch size

- **BS:**
  Global Batch size

- **TP:**
  Tensor parallel (1, 2, 4, 8)

- **SEQ_LENGTH**:
  Sequence Length

--- 

That's it! You've now set up the environment and configured the necessary settings for training Llama2 or Llama3 models.
