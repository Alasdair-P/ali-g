#!/bin/bash

#SBATCH --partition=small

# set the number of nodes
#SBATCH --nodes=1                                                                                                          

# set max wallclock time
#SBATCH --time=144:00:00

# set name of job
#SBATCH --job-name=cross_val

# set number of GPUs
#SBATCH --gres=gpu:1

# send mail to this address
#SBATCH --mail-user="alasdair.paren@gmail.com"

# run the application
./launch.sh
