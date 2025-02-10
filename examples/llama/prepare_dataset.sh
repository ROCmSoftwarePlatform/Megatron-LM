TMP_DIR="tmp"
mkdir -p $TMP_DIR
mkdir -p ${TMP_DIR}/data


MODEL_NAME="llama2"
TOKENIZER_MODEL_PATH=https://huggingface.co/NousResearch/Llama-2-7b-chat-hf/resolve/main/tokenizer.model
DATA_PATH="${TMP_DIR}/data"

usage() {
    echo "Usage: $0 --model-name <model name> --data-path <data-path>"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model-name) MODEL_NAME="$2"; shift ;;
        --data-path) DATA_PATH="$2"; shift ;;
        *) echo "unknown parameter passed: $1"; usage ;;
    esac
    shift
done

echo "MODEL NAME : $MODEL_NAME"
echo "DATA PATH NAME : $DATA_PATH"

TOKENIZER_MODEL=${TMP_DIR}/tokenizer.model

if [[ $MODEL_NAME == "llama2" ]]; then
    TOKENIZER_MODEL_PATH=https://huggingface.co/NousResearch/Llama-2-7b-chat-hf/resolve/main/tokenizer.model
elif [[ $MODEL_NAME == "llama3" ]]; then
    TOKENIZER_MODEL_PATH=meta-llama/Llama-3.1-8B
else
    echo "Unsupported model name for dataset preparation - use --model-name as llama2/llama3"
    exit 1
fi

# Download the tokenizer model
if ! [ -f "$TOKENIZER_MODEL" ]; then
    wget -O $TOKENIZER_MODEL $TOKENIZER_MODEL_PATH 
fi

python3 prepare_bookcorpus_megatron_dataset.py --out-dir ${DATA_PATH}
python3 ../../tools/preprocess_data.py --input ${DATA_PATH}/bookcorpus_megatron.json  --tokenizer-type GPTSentencePieceTokenizer \
--tokenizer-model ${TOKENIZER_MODEL} --output-prefix ${DATA_PATH}/bookcorpus --workers `nproc` --split-sentences

python3 ../../tools/preprocess_data.py --input ${DATA_PATH}/bookcorpus_megatron.json  --tokenizer-type GPTSentencePieceTokenizer \
--tokenizer-model ${TOKENIZER_MODEL} --output-prefix ${DATA_PATH}/bookcorpus --workers `nproc` --split-sentences
