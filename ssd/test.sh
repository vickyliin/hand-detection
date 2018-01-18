#!/bin/sh
export PYTHONPATH=$PYTHONPATH:`pwd`/research:`pwd`/research/slim
model_dir="research/object_detection/htc_real"

python3.6 research/object_detection/hand_detect.py 
