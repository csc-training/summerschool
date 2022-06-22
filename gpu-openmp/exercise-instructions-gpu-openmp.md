# General exercise instructions for openmp offloading to GPUs on Puhti

For most of the exercises, skeleton codes are provided both for
Fortran and C/C++ in the corresponding subdirectory. Some exercise
skeletons have sections marked with “TODO” for completing the
exercises. In addition, all of the 
exercises have exemplary full codes (that can be compiled and run) in the
`solutions` folder. Note that these are seldom the only or even the best way to
solve the problem.

## Compilation

Programs with OpenMP offloading can be build in Mahti with [NVIDIA HPC
Toolkit](https://docs.nvidia.com/hpc-sdk/index.html). The compiler
environment is enabled via module system:
```bash
module load nvhpc/21.2 ....
```
The compiler commands (without MPI) for C, C++ and Fortran are `nvc`,
`nvc++`, and `nvfortran`, and OpenMP offload support is enabled with
`-mp=gpu -gpu=cc70` options, *i.e.*

```
nvc -o my_exe test.c -mp=gpu -gpu=cc70
```
or
```
nvc++ -o my_exe test.cpp -mp=gpu -gpu=cc70
```
or
```
nvfortran -o my_exe test.f90 -mp=gpu -gpu=cc70
```


For MPI codes, use the wrapper commands `mpicc`, `mpic++`, or `mpif90`

## Running in Puhti

In Mahti, programs need to be executed via the batch job system. The
number of nodes is specified with `--nodes` (for most of the exercises
you should use only a single node), number of MPI tasks **per node**
with `--ntasks-per-node` (for exercises with single GPU this should be
one), and the number of GPUs per node with `--gres=gpu:a100:n`. If
program uses OpenMP with CPUs
number of cores reserved for threading is set with `--cpus-per-task`. The
actual number of threads is specified with `OMP_NUM_THREADS`
environment variable. Simple job running with single GPU can be
submitted with the following batch job script: 

```
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=<project>
#SBATCH --reservation=<reservation>
#SBATCH --partition=gpu
#SBATCH --reservation=openmp_offload
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1

srun my_exe
```

Save the script *e.g.* as `job.sh` and submit it with `sbatch job.sh`. The [exercises](exercises) 
directory contains additional job script templates (`job_xxx.sh`).
The output of job will be in file `slurm-xxxxx.out`. You can check the status of your jobs with `squeue -u $USER` and kill possible hanging applications with
`scancel JOBID`.

The reservation `openmp_offload` is available during the course days and it
is accessible only with the training user accounts.
