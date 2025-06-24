# General exercise instructions for LUMI

## Accessing LUMI

You can access the [LUMI](https://docs.lumi-supercomputer.eu/) supercomputer with ssh:

    ssh <username>@lumi.csc.fi

If you don't have the ssh key in the default location, you need to give a path to the key file:

    ssh -i <path-to-private-key> <username>@lumi.csc.fi

See [wiki](../../wiki/Setting-up-CSC-account-and-SSH) for further details.

## First-time setup on LUMI

All the exercises should be carried out in the scratch disk area.
This scratch area is shared between all the project members, so create a personal working directory there:

    mkdir -p /scratch/project_462000956/$USER
    cd /scratch/project_462000956/$USER

and clone the summer school git repository there:

    git clone https://github.com/csc-training/summerschool.git /scratch/project_462000956/$USER/summerschool

Now, `/scratch/project_462000956/$USER/summerschool` is your own clone of the summer school repository on LUMI
and you can modify files there without causing conflicts with other summer school participants.

<details>
<summary>Optional: Working with through your own git fork</summary>

**We don't have time to teach git during the summer school, so this is recommended only if you are already somewhat familiar with git.**

It is recommended to fork the summer school repository on github and clone your own fork to LUMI instead.
This allows you to push your exercise solutions to a branch on your own fork.

In order to push commits to your own fork on LUMI, you can use your existing key on your laptop by enabling ssh agent forwarding (`ssh -A <username>@lumi.csc.fi`) *or* to add an SSH public key generated on LUMI to your github account.

Note that the default editor for commit messages is *vim*, if you prefer something else you can add, e.g.,

    export EDITOR=nano

to the file `$HOME/.bashrc`.

</details>

## Using local workstation

In case you have working parallel program development environment in your laptop
(Fortran or C/C++ compiler, MPI development library, etc.) you may use that for
exercises. Note, however, that not much support for installing MPI environment or ROCM can be
provided during the course. Otherwise, you can use CSC supercomputers for
carrying out the exercises.


## Editors

For editing program source files you can use e.g. the *nano* editor:

    nano prog.f90

(`^` in nano's shortcuts refer to **Ctrl** key, *i.e.* in order to save the file and exit the editor press `Ctrl+X`)
Also other popular editors such as *emacs* and *vim* are available.


## Compilation

LUMI has several programming environments. For the summer school, we recommend that you use the Cray tools.

For CPU programming use:
```bash
module load LUMI/24.03
module load partition/C
```
For GPU programming use (except [SYCL](./exercise-instructions.md#sycl)):
```bash
module load LUMI/24.03
module load partition/G
module load rocm/6.0.3
```

### MPI

Compilation of the MPI programs can be performed with the `CC`, `cc`, or `ftn`
wrapper commands:
```
CC -o my_mpi_exe test.cpp
```
or
```
cc -o my_mpi_exe test.c
```
or
```
ftn -o my_mpi_exe test.f90
```

The wrapper commands include automatically all the flags needed for building MPI programs.

### OpenMP (threading with CPUs)

Pure OpenMP (as well as serial) programs can also be compiled with the `CC`,
`cc`, and `ftn` wrapper commands. OpenMP is enabled with the
`-fopenmp` flag:
```
CC -o my_exe test.cpp -fopenmp
```
or
```
cc -o my_exe test.c -fopenmp
```
or
```
ftn -o my_exe test.f90 -fopenmp
```

When the code also uses MPI, the wrapper commands include automatically all the flags needed for
building MPI programs.

### HDF5

In order to use HDF5 in CSC supercomputers, you need the load the HDF5 module with MPI I/O support.
The appropriate module in **Lumi** is
```
module load cray-hdf5-parallel
```

No special flags are needed for compiling and linking, the compiler wrappers take care of them automatically.


### OpenMP offloading

On **Lumi**, OpenMP offloading works with `PrgEnv-cray` and `PrgEnv-amd` programming environments. Otherwise load the GPU programming modules:

```bash
module load PrgEnv-cray
module load LUMI/24.03
module load partition/G
module load rocm
```

On **Lumi**, to compile your program, use
```bash
CC -fopenmp <source.cpp>
```

### HIP

Use the GPU programming modules:

```bash
module load PrgEnv-cray
module load LUMI/24.03
module load partition/G
module load rocm
```

To compile your program, use:
```bash
CC -xhip  <source.cpp>
```
HIP codes can be compiled as well using the `hipcc` AMD compiler:
```
hipcc --offload-arch=gfx90a  `CC --cray-print-opts=cflags` <source>.cpp `CC --cray-print-opts=libs` 
```
The flag `--offload-arch=gfx90a` indicates that we are targeting MI200 GPUs. If the code uses some libraries we need to extract them from the `CC` wrappers. This is done via the flags `CC --cray-print-opts=cflags` and  `CC --cray-print-opts=libs`.

A more elegant solution would be to use:
```
export HIPCC_COMPILE_FLAGS_APPEND="--offload-arch=gfx90a $(CC --cray-print-opts=cflags)"
export HIPCC_LINK_FLAGS_APPEND=$(CC --cray-print-opts=libs)

hipcc <source.cpp>
```
This is helpful when using make.

### SYCL
#### OneAPI + AMD Plug-in
Set-up the modules and paths:
```
source /projappl/project_462000956/apps/intel/oneapi/setvars.sh --include-intel-llvm
module load LUMI/24.03
module load partition/G
module load rocm/6.2.2
export  HSA_XNACK=1 # enables managed memory
export MPICH_GPU_SUPPORT_ENABLED=1                                # Needed for using GPU-aware MPI
```
Note that the intel initialization is done before loading the other modules to avoid overwriting the environment variables. 

Compile with:
```
icpx -fuse-ld=lld -std=c++20 -O3 -fsycl -fsycl-targets=amdgcn-amd-amdhsa,spir64_x86_64 -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a <sycl_code>.cpp
```
#### AdaptiveCPP
Set-up the modules and paths:
```
module load LUMI/24.03
module load partition/G
module load rocm/6.2.2
export PATH=/projappl/project_462000956/apps/ACPP/bin/:$PATH
export LD_LIBRARY_PATH=/appl/lumi/SW/LUMI-24.03/G/EB/Boost/1.83.0-cpeGNU-24.03/lib64/:$LD_LIBRARY_PATH
export LD_PRELOAD=/appl/lumi/SW/LUMI-24.03/G/EB/rocm/6.2.2/llvm/lib/libomp.so
export  HSA_XNACK=1 # enables managed memory
export MPICH_GPU_SUPPORT_ENABLED=1                                # Needed for using GPU-aware MPI
``` 
Compile with:
```
acpp -O3 --acpp-targets="omp.accelerated;hip:gfx90a" <sycl_code>.cpp
```

## Running in LUMI

#### Pure MPI

Programs need to be executed via the batch job system. A simple job running with 4 MPI tasks can be submitted with the following batch job script:
```
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_462000956
#SBATCH --partition=small
#SBATCH --reservation=CSC_summer_school_cpu
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1

srun ./my_mpi_exe
```

Save the script *e.g.* as `job.sh` and submit it with `sbatch job.sh`.
The output of the job will be in the file `slurm-xxxxx.out`. You can check the status of your jobs with `squeue -u $USER` and kill possible hanging applications with `scancel JOBID`.

The reservation `summerschool` is available during the course days and it is accessible only with the training user accounts.

The same result can be achieved using directly `srun`
```
srun --job-name=example --account=project_462000956 --partition=small --reservation=CSC_summer_school_cpu --time=00:05:00 --nodes=1 --ntasks-per-node=4 --cpus-per-task=1  my_mpi_exe
```
#### Pure OpenMP

For pure OpenMP programs one should use only one node and one MPI task per node and specify the number of cores reserved
for threading with `--cpus-per-task`:
```
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_462000956
#SBATCH --partition=small
#SBATCH --reservation=CSC_summer_school_cpu
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

# Set the number of threads based on --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun ./my_omp_exe
```

The same result can be achieved using directly `srun`
```
srun --job-name=example --account=project_462000956 --partition=small --reservation=CSC_summer_school_cpu --time=00:05:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=4   my_omp_exe
```
#### Hybrid MPI+OpenMP

For hybrid MPI+OpenMP programs it is recommended to specify explicitly the number of nodes, the number of
MPI tasks per node (pure OpenMP programs as special case with one node and one task per node),
and the number of cores reserved for threading. The number of nodes is specified with `--nodes`
(for most of the exercises you should use only a single node), the number of MPI tasks **per node**
with `--ntasks-per-node`, and the number of cores reserved for threading with `--cpus-per-task`.
The actual number of threads is specified with the `OMP_NUM_THREADS` environment variable.
A simple job running with 32 MPI tasks and 4 OpenMP threads per MPI task can be submitted with
the following batch job script:
```
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_462000956
#SBATCH --partition=small
#SBATCH --reservation=CSC_summer_school_cpu
#SBATCH --time=00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=4
# Set the number of threads based on --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun ./my_exe
```

The same result can be achieved using directly `srun`
```
srun --job-name=example --account=project_462000956 --partition=small --reservation=CSC_summer_school_cpu --time=00:05:00 --nodes=2 --ntasks-per-node=32 --cpus-per-task=4   my_omp_exe
```
#### GPU programs

When running GPU programs, few changes need to made to the batch job
script. The `partition` is now different, and one must also request explicitly a given number of GPUs per node with the
`--gpus-per-node=X` option. As an example, in order to use a
single GPU with single MPI task and a single thread use:
```
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_462000956
#SBATCH --partition=small-g
#SBATCH --reservation=CSC_summer_school_gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00

srun ./my_gpu_exe
```

The same result can be achieved using directly `srun`
```
srun --job-name=example --account=project_462000956 --partition=small-g --reservation=CSC_summer_school_gpu --time=00:05:00 --gpus-per-node=1 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1  my_gpu_exe
```
**Note!** Some programs require GPU-aware MPI to perform MPI operations using directly the GPU pointers (this is avoiding some GPU-CPU transfers). This is enabled via:

```
export MPICH_GPU_SUPPORT_ENABLED=1
```

#### Interactive jobs


When debugging or doing performance analysis the user needs to interact with the application on the compute nodes.

```bash
salloc --account=<project_id> –-partition=small –-nodes=2 –-ntasks-per-nodes=128 --time=00:30:00
```
Once the allocation is made, this command will start a shell on the login node.

```bash
srun --ntasks=32 --cpus-per-task=8 ./my_interactive_prog
```

## Debugging

See [the MPI debugging exercise](mpi/debugging),
[CSC user guide](https://docs.csc.fi/computing/debugging/), and
[LUMI documentation](https://docs.lumi-supercomputer.eu/development/)
for possible debugging options.
