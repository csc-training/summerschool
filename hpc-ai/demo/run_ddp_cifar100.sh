#!/bin/bash
#SBATCH --account=project_462000365
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --mem=60G
#SBATCH --time=0:30:00

module use /appl/local/csc/modulefiles/
module load pytorch/2.4

srun torchrun --nproc_per_node=2 train_ddp_cifar100.py
