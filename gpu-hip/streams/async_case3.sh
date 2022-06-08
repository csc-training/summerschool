#!/bin/bash
#SBATCH --job-name=async_serial
#SBATCH --time=00:05:00
#SBATCH --partition=MI100
#SBATCH --nodes=1
#SBATCH -w ixt-sjc2-69
#sbatch --reservation=lumi

export ROCR_VISIBLE_DEVICES=0
srun -n 1 ./async_case3
