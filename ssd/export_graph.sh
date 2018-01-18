#!/bin/sh
export PYTHONPATH=$PYTHONPATH:`pwd`/research:`pwd`/research/slim
model_dir="research/object_detection/htc_real"
rm -r $model_dir/output

python3.6 research/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path $model_dir/ssd_mobilenet_v1.config \
    --trained_checkpoint_prefix $model_dir/train_logs/model.ckpt-50000 \
    --output_directory $model_dir/output/
