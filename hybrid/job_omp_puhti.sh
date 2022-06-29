#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_2000745
#SBATCH --partition=large
#SBATCH --reservation=summerschool
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

srun my_omp_exe
