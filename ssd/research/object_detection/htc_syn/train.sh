mkdir -p logs/
now=$(date +"%Y%m%d_%H%M%S")
python3.6 ../../train.py \
    --logtostderr \
    --pipeline_config_path=ssd_mobilenet_v1.config \
    --train_dir=train_logs 2>&1 | tee logs/train_$now.txt &
