#!/bin/bash
#SBATCH --job-name=beer
#SBATCH --account=project_465000536
#SBATCH --error=%x.%J.err
#SBATCH --output=%x.%J.out
#SBATCH --partition=standard-g
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
##SBATCH --reservation=

srun prog
