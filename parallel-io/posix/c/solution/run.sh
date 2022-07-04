#!/bin/bash -x
#SBATCH --account=Project_2002078
#SBATCH --partition=small
#SBATCH --reservation=summerschool
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=log.out
#SBATCH --error=log.err

# srun --account=Project_2002078 --partition=small --reservation=summerschool --ntasks=1 --cpus-per-task=1 --time=00:15:00 a.out

srun a.out 4

### 1 Run vtune analysis
# rm -rf vtune_hotspots
# source /appl/opt/testing/intel-oneapi/setvars.sh --force
# srun amplxe-cl -r vtune_hotspots -collect hotspots -- ./laplacian

### 2 Launch GUI
# vtune-gui results_dir_name

### Compile with g++
# g++ -g -O2 -fopenmp -fopt-info-loop -o laplacian laplacian.cpp
# g++ -g -O3 -fopenmp -fopt-info-loop -o laplacian_matrix laplacian_matrix.cpp
