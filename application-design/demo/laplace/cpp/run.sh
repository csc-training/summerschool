#!/bin/bash -x
#SBATCH --account=Project_2002078
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:15:00
#SBATCH --output=log.out
#SBATCH --error=log.err

# srun laplace 4
srun nsys profile -o nsys_prof --stats=false --trace=cuda --force-overwrite=true ./laplace
# srun ncu -o ncu_prof --force-overwrite --set full --kernel-regex hipKernel --launch-skip 0 --launch-count 1 ./laplace
