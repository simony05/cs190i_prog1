#!/bin/bash

#SBATCH --job-name=sbatch_example
#SBATCH -D .
#SBATCH --output=/cs/student/simonryan/cs190i/cs190i_prog1/sbatch/O-%x.%j
#SBATCH --error=/cs/student/simonryan/cs190i/cs190i_prog1/sbatch/E-%x.%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # number of MP tasks
#SBATCH --gres=gpu:1 # number of GPUs per node
#SBATCH --time=02:00:00 # maximum execution time (HH:MM:SS)
#SBATCH --cpus-per-task=4

######################
### Set enviroment ###
######################

cd /cs/student/simonryan/cs190i/cs190i_prog1/split7_box2
python train.py

