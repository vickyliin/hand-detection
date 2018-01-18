#!/bin/sh
export PYTHONPATH=$PYTHONPATH:`pwd`/research:`pwd`/research/slim
now=$(date +"%Y%m%d_%H%M%S")
model_dir="research/object_detection/htc_syn"
mkdir -p $model_dir/logs/

python3.6 research/object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=$model_dir/ssd_mobilenet_v1.config \
    --train_dir=$model_dir/train_logs 2>&1 | tee $model_dir/logs/train_$now.txt


model_dir="research/object_detection/htc_real"
mkdir -p $model_dir/logs/
python3.6 research/object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=$model_dir/ssd_mobilenet_v1.config \
    --train_dir=$model_dir/train_logs 2>&1 | tee $model_dir/logs/train_$now.txt
