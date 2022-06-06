#!/bin/bash
#SBATCH --job-name=matrix_transpose_naive
#SBATCH --time=00:05:00
#SBATCH --partition=MI100
#SBATCH --nodes=1
#SBATCH -w ixt-sjc2-69
#sbatch --reservation=lumi

export ROCR_VISIBLE_DEVICES=0
srun -n 1 rocprof -i metrics_matrix_transpose_naive_kernel.txt -o metrics_naive.csv ./matrix_transpose_naive
