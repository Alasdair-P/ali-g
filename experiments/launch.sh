#!/bin/bash

module load python3/anaconda
module load cuda/9.0
source activate torch_env
export VISION_DATA="/jmain01/home/JAD035/pkm01/lxb12-pkm01/data/vision/datasets"

nvidia-smi
echo "Using CUDA device" $CUDA_VISIBLE_DEVICES

python scripts/cv_svhn.py
