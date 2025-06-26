# General exercise instructions for Mahti

## Accessing Mahti

You can access the [Mahti](https://docs.csc.fi/computing/systems-mahti/) supercomputer with ssh:

    ssh <username>@mahti.csc.fi

If you don't have the ssh key in the default location, you need to give a path to the key file:

    ssh -i <path-to-private-key> <username>@mahti.csc.fi

See [wiki](../../wiki/Setting-up-CSC-account-and-SSH) for further details.

## First-time setup on Mahti

All the exercises should be carried out in the scratch disk area.
This scratch area is shared between all the project members, so create a personal working directory there:

    mkdir -p /scratch/project_2014370/$USER
    cd /scratch/project_2014370/$USER

and clone the summer school git repository there:

    git clone https://github.com/csc-training/summerschool.git /scratch/project_2014370/$USER/summerschool

Now, `/scratch/project_2014370/$USER/summerschool` is your own clone of the summer school repository on Mahti
and you can modify files there without causing conflicts with other summer school participants.

<details>
<summary>Optional: Working with your own git fork</summary>

**We don't have time to teach git during the summer school, so this is recommended only if you are already somewhat familiar with git.**

It is recommended to fork the summer school repository on github and clone your own fork to Mahti instead.
This allows you to push your exercise solutions to a branch on your own fork.

In order to push commits to your own fork on Mahti, you can use your existing key on your laptop by enabling ssh agent forwarding (`ssh -A <username>@mahti.csc.fi`) *or* to add an SSH public key generated on Mahti to your github account.

Note that the default editor for commit messages is *vim*, if you prefer something else you can add, e.g.,

    export EDITOR=nano

to the file `$HOME/.bashrc`.

</details>


## Compilation

### MPI

Compilation of the MPI programs can be performed with the `mpif90`,
`mpicxx`, and `mpicc` wrapper commands:
```
mpif90 -o my_mpi_exe test.f90
```
or
```
mpicxx -o my_mpi_exe test.cpp
```
or
```
mpicc -o my_mpi_exe test.c
```

The wrapper commands include automatically all the flags needed for building
MPI programs.

### OpenMP (threading with CPUs)

Pure OpenMP (as well as serial) programs can also be compiled with the `mpif90`,
`mpicxx`, and `mpicc` wrapper commands. OpenMP is enabled with the
`-fopenmp` flag:
```
mpif90 -o my_exe test.f90 -fopenmp
```
or
```
mpicxx -o my_exe test.cpp -fopenmp
```
or
```
mpicc -o my_exe test.c -fopenmp
```

When code uses also MPI, the wrapper commands include automatically all the flags needed for
building MPI programs.

### HDF5

In order to use HDF5 in CSC supercomputers, you need the load the HDF5 module with MPI I/O support.
The appropriate module in Mahti is
```
module load hdf5/1.10.7-mpi
```

When building programs, `-lhdf5` (C/C++) or `-lhdf5_fortran` (Fortran) needs to be added to linker flags, e.g.
```
mpicxx -o my_hdf5_exe test.cpp -lhdf5
mpif90 -o my_hdf5_exe test.f90 -lhdf5_fortran
```
or setting `LDFLAGS` *etc.* in a Makefile:
```
LDFLAGS=... -lhdf5
```

Usage in local workstation may vary.

### OpenMP offloading

On **Mahti**, in order to use programs with OpenMP offloading to GPUs, you need to reconfigure the modulepaths as follows:
```bash
module purge
module use /appl/opt/nvhpc/modulefiles
```

Please note that this modification has implications on the consistency of the module tree, see CSC's user [documentation](https://docs.csc.fi/computing/compiling-mahti/#openacc-and-openmp-offloading) for more information.

After this change you can load the Nivida nvhpc module:
``` bash
module load nvhpc-hpcx-cuda12/25.1
```

On **Mahti**, the compiler commands (without MPI) for C, C++ and Fortran are `nvc`,
`nvc++`, and `nvfortran`, and OpenMP offload support is enabled with
`-mp=gpu -gpu=cc80` options, *i.e.*

```
nvc -o my_exe test.c -mp=gpu -gpu=cc80
```
or
```
nvc++ -o my_exe test.cpp -mp=gpu -gpu=cc80
```
or
```
nvfortran -o my_exe test.f90 -mp=gpu -gpu=cc80
```

For MPI codes, use the wrapper commands `mpicc`, `mpic++`, or `mpif90`

### HIP

In order to use HIP on **Mahti**, you need to reconfigure the module paths as follows:
```
module purge
module use /projappl/project_2014370/spack-container/modules/Core
```

After this you can load the following modules:
``` bash
module load gcc/13.3.0 hip cuda
```

Then you can compile with hipcc, eg,
```
hipcc  --gpu-architecture=sm_80 -o hello hello.cpp
```
where `--gpu-architecture=sm_80` is required when compiling for A100.

## Running in Mahti

### Pure MPI

In Mahti, programs need to be executed via the batch job system. A job running with 4 MPI tasks can be submitted with the following batch job script:
```
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_2014370
#SBATCH --partition=small
#SBATCH --time=00:05:00
#SBATCH --ntasks=4

srun ./my_mpi_exe
```

Save the script *e.g.* as `job.sh` and submit it with `sbatch job.sh`.
The output of job will be in file `slurm-xxxxx.out`. You can check the status of your jobs with `squeue -u $USER` and kill possible hanging applications with
`scancel JOBID`.

### Pure OpenMP

For pure OpenMP programs one should use only single tasks and specify the number of cores reserved
for threading with `--cpus-per-task`.
```
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_2014370
#SBATCH --partition=small
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

srun ./my_omp_exe
```

### Hybrid MPI+OpenMP

For hybrid MPI+OpenMP programs it is recommended to specify explicitly number of nodes, number of
MPI tasks per node (pure OpenMP programs as special case with one node and one task per node),
and number of cores reserved for threading. The number of nodes is specified with `--nodes`
(for most of the exercises you should use only a single node), number of MPI tasks **per node**
with `--ntasks-per-node`, and number of cores reserved for threading with `--cpus-per-task`.
The actual number of threads is specified with `OMP_NUM_THREADS` environment variable.
Simple job running with 4 MPI tasks and 4 OpenMP threads per MPI task can be submitted with
the following batch job script:
```
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_2014370
#SBATCH --partition=medium
#SBATCH --time=00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=4

# Set the number of threads based on --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun ./my_exe
```

When using only single node, one should use the `small` partition, *i.e.*
```
...
#SBATCH --partition=small
SBATCH --nodes=1
...
```

### GPU programs

When running GPU programs, few changes need to made to the batch job
script. The `partition` is now different, and one
must also request explicitly given number of GPUs with the
`--gres=gpu:a100:ngpus` option. As an example, in order to use a
single GPU with single MPI task and a single thread use:
```
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_2014370
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:05:00

srun ./my_gpu_exe
```
