#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_462000956
##SBATCH --reservation=CSC_summer_school_gpu
#SBATCH --output=%x.%J.out
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --time=00:05:00

# Enable GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

srun ./prog
