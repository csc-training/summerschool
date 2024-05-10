#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_465zzzzz
#SBATCH --error=%x.%J.err
#SBATCH --output=%x.%J.out
#SBATCH --partition=standard
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
##SBATCH --reservation=

srun my_prog
