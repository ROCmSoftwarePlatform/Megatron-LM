mkdir -p tuned/qwen1.5_14b
mv rocblas.yaml tuned/qwen1.5_14b/
mv full_tuned*.csv tuned/qwen1.5_14b/

export PYTORCH_TUNABLEOP_FILENAME=tuned/qwen1.5_14b/full_tuned%d.csv
export PYTORCH_TUNABLEOP_TUNING=0
export PYTORCH_TUNABLEOP_ENABLED=1
bash train_qwen1.5_14b.sh
