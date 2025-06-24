#!/bin/bash
#SBATCH --job-name=scalability
#SBATCH --account=project_462000956
#SBATCH --reservation=SummerSchoolGPU
#SBATCH --output=%x.%J.out
#SBATCH --partition=small-g
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1
#SBATCH --time=00:05:00

# Enable GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

srun ./heat_hip
