# General exercise instructions

For most of the exercises, skeleton codes are provided both for
Fortran and C/C++ in the corresponding subdirectory. Some exercise
skeletons have sections marked with “TODO” for completing the
exercises. In addition, all of the
exercises have exemplary full codes (that can be compiled and run) in the
`solutions` folder. Note that these are seldom the only or even the best way to
solve the problem.

The exercise material can be downloaded with the command

```
git clone https://github.com/csc-training/summerschool.git
```

However, we recommend that you use your GitHub account (and create a one if not having yet),
**Fork** this repository and clone then your fork. This way you can keep also your own work
under version control.

## Using local workstation

In case you have working parallel program development environment in your laptop
(Fortran or C/C++ compiler, MPI development library, etc.) you may use that for
exercises. Note, however, that no support for installing MPI environment can be
provided during the course. Otherwise, you can use CSC supercomputers for
carrying out the exercises.

## Using CSC supercomputers

Exercises can be carried out using the CSC's [LUMI](https://docs.lumi-supercomputer.eu/)  supercomputer.

LUMI can be accessed via ssh using the provided username and ssh key pair:
```
ssh -i <path-to-private-key> <username>@lumi.csc.fi

```

For editing program source files you can use e.g. *nano* editor:

```
nano prog.f90 &
```
(`^` in nano's shortcuts refer to **Ctrl** key, *i.e.* in order to save file and exit editor press `Ctrl+X`)
Also other popular editors (emacs, vim, gedit) are available.

### Disk areas

All the exercises in the supercomputers should be carried out in the
**scratch** disk area. The name of the scratch directory can be
queried with the command `lumi-workspaces`. As the base directory is
shared between members of the project, you should create your own
directory:
```
cd /scratch/project_465000536/
mkdir -p $USER
cd $USER
```


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
The appropriate module in Puhti is
```
module load hdf5/1.10.4-mpi
```
and in Mahti
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

On **Lumi**, the following modules are required:

```bash
module load LUMI/22.08
module load partition/G 
module load rocm/5.3.3
```

On **Lumi**, to compile your program, use
```bash
CC -fopenmp <source.cpp>
```


### HIP

On **Lumi**, the following modules are required:

```bash
module load LUMI/22.08
module load partition/G
module load rocm/5.3.3
```

On **Lumi**, to compile your program, use
```bash
CC -xhip <source.cpp>
```

## Running in Puhti

### Pure MPI

Programs need to be executed via the batch job system. Simple job running with 4 MPI tasks can be submitted with the following batch job script:
```
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_465000536
#SBATCH --partition=standard
#SBATCH --time=00:05:00
#SBATCH --ntasks=4
.....!
srun my_mpi_exe
```

Save the script *e.g.* as `job.sh` and submit it with `sbatch job.sh`.
The output of job will be in file `slurm-xxxxx.out`. You can check the status of your jobs with `squeue -u $USER` and kill possible hanging applications with
`scancel JOBID`.

The reservation `summerschool` is available during the course days and it
is accessible only with the training user accounts.

### Pure OpenMP

For pure OpenMP programs one should use only single tasks and specify the number of cores reserved
for threading with `--cpus-per-task`. Furthermore, one should use the `small` partition:
```
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_465000536
#SBATCH --partition=standard
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
......!
srun my_omp_exe
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
#SBATCH --account=project_465000536
#SBATCH --partition=standard
#SBATCH --time=00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=4
.....!
# Set the number of threads based on --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun my_exe
```


### GPU programs

When running GPU programs, few changes need to made to the batch job
script. The `partition` and `reservation` are now different, and one
must also request explicitly given number of GPUs with the
`--gres=gpu:v100:ngpus` option. As an example, in order to use a
single GPU with single MPI task and a single thread use:
```
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_465000536
#SBATCH --partition=standard-g
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00

srun my_gpu_exe
```

## Running in local workstation

In most MPI implementations parallel program can be started with the `mpiexec` launcher:
```
mpiexec -n 4 ./my_mpi_exe
```

In most workstations, programs build with OpenMP use as many threads as there are CPU cores
(note that this might include also "logical" cores with simultaneous multithreading). A pure OpenMP
program can be normally started with specific number of threads with
```bash
OMP_NUM_THREADS=4 ./my_exe
```
and a hybrid MPI+OpenMP program e.g. with
```
OMP_NUM_THREADS=4 mpiexec -n 2 ./my_exe
```

## Debugging in CSC supercomputers

## Performance analysis with TAU and Omniperf

More information about TAU can be found in [TAU User Documentation](https://docs.csc.fi/apps/scalasca/), while for Omniperf at [Omniperf User Documentation](https://docs.csc.fi/apps/scalasca/)
