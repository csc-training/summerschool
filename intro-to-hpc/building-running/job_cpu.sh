#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_462000956
#SBATCH --reservation=SummerSchoolCPU
#SBATCH --output=%x.%J.out
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:05:00

srun ./prog
