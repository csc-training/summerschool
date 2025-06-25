#!/bin/bash
#SBATCH --account=project_462000365
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=60G
#SBATCH --time=0:30:00

module use /appl/local/csc/modulefiles/
module load pytorch/2.4

srun python3 train_cifar100.py
