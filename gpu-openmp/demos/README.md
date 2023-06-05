Commands to compile and run this demo on LUMI

CC -fopenmp vector.cpp
srun --account=<project_id> -n 1 --ntasks-per-node=1 --partition=small-g --gres=gpu:mi250:1 a.out
