#!/bin/bash -l

#SBATCH --account=project_465001194
#SBATCH --job-name=01_three_kernels
#SBATCH --output=01_three_kernels.out%j
#SBATCH --error=01_three_kernelsname.err%j
#SBATCH --partition=small-g
#SBATCH --reservation=CSC_summer_school_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=10G
#SBATCH --time=00:30:00

ml LUMI/23.09
ml partition/G
ml PrgEnv-cray
ml craype-accel-amd-gfx90a
ml rocm/5.4.6

export PATH=/projappl/project_465001194/apps/omniperf/bin:$PATH

srun omniperf profile -n 01_three_kernels -- ./a.out
