#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_2000745
#SBATCH --partition=medium
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun my_exe
