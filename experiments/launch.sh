#!/bin/bash

module purge
module load python3/anaconda
module load cuda/9.0
source activate /jmain01/home/JAD035/pkm01/aap21-pkm01/anaconda3/envs/myenv
export VISION_DATA="/jmain01/home/JAD035/pkm01/shared/datasets"

nvidia-smi
echo "Using CUDA device" $CUDA_VISIBLE_DEVICES

python reproduce/cifar_jade.py
# python test.py
