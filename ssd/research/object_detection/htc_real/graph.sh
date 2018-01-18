#!/bin/sh
python3.6 ../../export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ssd_mobilenet_v1.config \
    --trained_checkpoint_prefix train_logs/model.ckpt-18893 \
    --output_directory output/
