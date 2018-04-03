mkdir -p eval_logs
python3.6 ../../eval.py \
    --logtostderr \
    --pipeline_config_path=ssd_mobilenet_v1.config \
    --checkpoint_dir=train_logs \
    --eval_training_data=True \
    --eval_dir=check_logs \
    --eval_interval_secs=180
