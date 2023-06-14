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

Exercises can be carried out using the CSC's Puhti and Mahti supercomputers. In this course,
we'll use by default Puhti. See CSC User Documentation for quick start guide for [Puhti](https://docs.csc.fi/support/tutorials/puhti_quick/) or for [Mahti](https://docs.csc.fi/support/tutorials/mahti_quick/)

Puhti and Mahti can be accessed via ssh using the
provided username (`trainingxxx`) and password:
```
ssh -Y training000@puhti.csc.fi
```
or
```
ssh -Y training000@mahti.csc.fi
```

For easier connecting we recommend that you set up *ssh keys* along the instructions in
[CSC Docs](https://docs.csc.fi/computing/connecting/#setting-up-ssh-keys)


For editing program source files you can use e.g. *nano* editor:

```
nano prog.f90 &
```
(`^` in nano's shortcuts refer to **Ctrl** key, *i.e.* in order to save file and exit editor press `Ctrl+X`)
Also other popular editors (emacs, vim, gedit) are available.

### Disk areas

All the exercises in the supercomputers should be carried out in the
**scratch** disk area. The name of the scratch directory can be
queried with the command `csc-workspaces`. As the base directory is
shared between members of the project, you should create your own
directory:
```
cd /scratch/project_2000745
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
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm/5.3.3
```

On **Lumi**, to compile your program, use
```bash
CC -fopenmp <source.cpp>
```

On **Puhti**, in order to use programs with OpenMP offloading to GPUs, you need to load the following modules:
```bash
module load .unsupported
module load nvhpc/22.7
```

On **Puhti**, the compiler commands (without MPI) for C, C++ and Fortran are `nvc`,
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

### HIP

In order to use HIP on Puhti, you need to load the following modules:
```
module load gcc/9.1.0 cuda/11.1.0 hip/4.0.0 openmpi/4.1.1-cuda
```
Then you can compile with hipcc, eg,
```
hipcc  --gpu-architecture=sm_70 -o hello hello.cpp
```
where `--gpu-architecture=sm_70` is required when compiling for V100.

## Running in Puhti

### Pure MPI

In Puhti, programs need to be executed via the batch job system. Simple job running with 4 MPI tasks can be submitted with the following batch job script:
```
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_2000745
#SBATCH --partition=large
#SBATCH --reservation=summerschool
#SBATCH --time=00:05:00
#SBATCH --ntasks=4

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
#SBATCH --account=project_2000745
#SBATCH --partition=small
#SBATCH --reservation=summerschool
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

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
#SBATCH --account=project_2000745
#SBATCH --partition=large
#SBATCH --reservation=summerschool
#SBATCH --time=00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=4

# Set the number of threads based on --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun my_exe
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
script. The `partition` and `reservation` are now different, and one
must also request explicitly given number of GPUs with the
`--gres=gpu:v100:ngpus` option. As an example, in order to use a
single GPU with single MPI task and a single thread use:
```
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_2000745
#SBATCH --partition=gpu
#SBATCH --reservation=summerschool-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:05:00

srun my_gpu_exe
```

## Running in Mahti

Batch job system in Mahti is very similar to Puhti, and batch scripts written for Puhti
work in Mahti with only minor modifications. The most important difference is that one should use
the `medium` partition instead of `large` partition, *i.e.* batch job script should start as:
```
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_2000745
#SBATCH --partition=medium
#SBATCH --reservation=summerschool
...
```

For GPU programs, the `--gres` option is also a bit different, as one
needs to use `a100` instead of `v100` *i.e.*:
```
...
#SBATCH --gres=gpu:a100:1
...
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

The [Allinea DDT parallel debugger](https://docs.csc.fi/apps/ddt/) is available in CSC
supercomputers. In order to use the debugger, build your code first with the `-g` flag. The DDT is
then enabled via the module system:

```bash
module load ddt
```

The debugger is run in an interactive session, and for proper
functioning the environment variable `SLURM_OVERLAP` needs to be set.

1. Set `SLURM_OVERLAP` and request Slurm allocation interactively:
```bash
export SLURM_OVERLAP=1
salloc --nodes=1 --ntasks-per-node=2 --account=project_2000745 --partition=small --reservation=mpi_intro
```
2. Start the application under debugger
```bash
ddt srun ./buggy
```

For GUI applications we recommend to use the
[Desktop app](https://docs.csc.fi/computing/webinterface/desktop/) in the Puhti web interface.
For smoother GUI performance one may use VNC client such as RealVNC or TigerVNC in the local
workstation.

## Performance analysis with ScoreP / Scalasca

Start by loading `scorep` and `scalasca` modules:

```bash
module load scorep scalasca
```

Instrument the application by prepeding compile command with `scorep`:

```bash
scorep mpicc -o my_mpi_app my_mpi_code.c
```

Collect and create flat profile by prepending `srun` with `scan`:
```
...
#SBATCH --ntasks=8

module load scalasca
scan srun ./my_mpi_app
```

Scalasca analysis report explorer `square` does not work currently in
the CSC supercomputers, but the experiment directory can be copied to
local workstation for visual analysis:

(On local workstation)
```bash
rsync -r puhti.csc.fi:/path_to_rundir/scorep_my_mpi_app_8_sum .
```

The `scorep-score` command can be used also in the supercomputers to
estimate storage requirements before starting tracing:

```bash
scorep-score -r scorep_my_mpi_app_8_sum/profile.cubex
```

In order to collect and analyze the trace, add `-q` and `-t` options
to `scan`:

```bash
...
#SBATCH --ntasks=8

module load scalasca
scan -q -t srun ./my_mpi_app
```

The experiment directory containing the trace can now be copied to
local workstation for visual analysis:

```bash
rsync -r puhti.csc.fi:/path_to_rundir/scorep_my_mpi_app_8_trace .
```

On CSC supercomputers, one can use Intel Traceanalyzer for
investigating the trace (Traceanalyzer can read the `.otf2` produced
by ScoreP / Scalasca):

```bash
module load intel-itac
traceanalyzer &
```

Next, choose the "Open" dialog and select the `trace.otf2` file within
the experiment directory (e.g. `scorep_my_mpi_app_8_trace`).
For GUI applications we recommend to use the
[Desktop app](https://docs.csc.fi/computing/webinterface/desktop/) in the Puhti web interface.
For smoother GUI performance one may use VNC client such as RealVNC or TigerVNC in the local
workstation.


More information about Scalasca can be found in [CSC User Documentation](https://docs.csc.fi/apps/scalasca/)
