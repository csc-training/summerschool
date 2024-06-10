Commands to compile and run this demo on LUMI

Load modules:
```
module load PrgEnv-cray
module load rocm
module load craype-accel-amd-gfx90a
```

Compile: 
> CC -fopenmp vector.cpp

Run:
> srun --account=<project_id> -n 1 -N 1 --partition=small-g -G 1 ./a.out

