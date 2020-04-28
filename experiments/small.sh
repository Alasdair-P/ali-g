#!/bin/bash

#SBATCH --partition=small

# set the number of nodes
#SBATCH --nodes=1                                                                                                          

# set max wallclock time
#SBATCH --time=72:00:00

# set name of job
#SBATCH --job-name=test

# set number of GPUs
#SBATCH --array=0-1

# send mail to this address
#SBATCH --mail-user="alasdair.paren@gmail.com"

# run the application
./launch.sh
