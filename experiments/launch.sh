#!/bin/bash

module purge
# module load python3/anaconda
module load cuda/9.2
{
	source activate /jmain01/home/JAD035/pkm01/aap21-pkm01/anaconda3/envs/tboard
	wait $!
	printf “sucess”
}||{
	printf “failed”
}
export VISION_DATA="/jmain01/home/JAD035/pkm01/shared/datasets"

nvidia-smi
echo "Using CUDA device" $CUDA_VISIBLE_DEVICES

# python reproduce/jade_cifar.py
python reproduce/svhn_jade.py
