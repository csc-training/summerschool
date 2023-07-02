# General exercise instructions

## Working with the summerschool repository

We recommend that you work with your own personal **fork** of the summerschool 
git-repository. That way you can easily commit and push your own solutions
to exercises.

Before starting out, synchronize your fork with "Sync fork" in the github web-GUI.

We also recommend that you create a separate branch for your own work, see "Using local workstation" or "Using supercomputers" below for details.

Once forked to yourself, you can sync with the original repository (in case of updates) by running:
```
git pull https://github.com/csc-training/summerschool.git
```

### Repository structure

The exercise assignments are provided in various `README.md`s.
For most of the exercises, skeleton codes are provided both for
Fortran and C/C++ in the corresponding subdirectory. Some exercise
skeletons have sections marked with “TODO” for completing the
exercises. In addition, all of the exercises have exemplary full codes 
(that can be compiled and run) in the `solutions` folder. Note that these are 
seldom the only or even the best way to solve the problem.

## Using local workstation

In case you have working parallel program development environment in your laptop
(Fortran or C/C++ compiler, MPI development library, etc.) you may use that for
exercises. Note, however, that no support for installing MPI environment can be
provided during the course. Otherwise, you can use CSC supercomputers for
carrying out the exercises.

Clone your personal fork in appropriate directory:
```
git clone git@github.com:<my-github-id>/summerschool.git
```
Create a branch:
```
git checkout -b hpcss23
```

## Using supercomputers

Exercises can be carried out using the [LUMI](https://docs.lumi-supercomputer.eu/)  supercomputer.

LUMI can be accessed via ssh using the provided username and ssh key pair:
```
ssh -i <path-to-private-key> <username>@lumi.csc.fi
```

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

### Working with git

In order to push code to your own fork, you need to add your SSH public key in LUMI to
your github account. The SSH key can be added via "Settings"->"SSH and GPG keys"->"New SSH key", by copy-pasting output of the following command:
```
cat $HOME/.ssh/id_rsa.pub
```

Once succesfull, make sure you in your personal workspace in **scratch** area `/scratch/project_465000536/$USER`, clone the repository, and a create a branch:
```
git clone git@github.com:<my-github-id>/summerschool.git
git checkout -b hpcss23
```

If you haven't used git before in LUMI, you need to add also your identity:
```
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
```

Default editor for commit messages is *vim*, if you prefer something else you can add
to the file `$HOME/.bashrc` *e.g.*
```
export EDITOR=nano
```

### Editors

For editing program source files you can use e.g. *nano* editor:

```
nano prog.f90
```
(`^` in nano's shortcuts refer to **Ctrl** key, *i.e.* in order to save file and exit editor press `Ctrl+X`)
Also other popular editors such as emacs and vim are available. 

## Compilation

LUMI has several programming environments. For summerschool, we recommend that you use
the special summerschool modules:
```
module use /project/project_465000536/modules
```
For CPU programming use:
```
module load hpcss/cpu
```
For GPU programming use:
```
module load hpcss/gpu
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

The wrapper commands include automatically all the flags needed for building
MPI programs.

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

When code uses also MPI, the wrapper commands include automatically all the flags needed for
building MPI programs.

### HDF5

In order to use HDF5 in CSC supercomputers, you need the load the HDF5 module with MPI I/O support.
The appropriate module in **Lumi** is
```
module load cray-hdf5-parallel/1.12.2.1
```

No special flags are needed for compiling and linking, the compiler wrappers take care of them automatically.

Usage in local workstation may vary.

### OpenMP offloading

On **Lumi**, the following modules are required:

```bash
module load LUMI/22.08
module load partition/G 
module load cce/15.0.1
module load rocm/5.3.3
```

On **Lumi**, to compile your program, use
```bash
CC -fopenmp <source.cpp>
```


### HIP

Use the following modules :

```bash
module load LUMI/22.08
module load partition/G
module load cce/15.0.1
module load rocm/5.3.3
```

To compile your program, use:
```bash
CC -xhip <source.cpp>
```
### HIPFORT 
The following modules are required:
```bash

module load LUMI/22.08
module load partition/G
module load cce/15.0.1
module load rocm/5.3.3
```

Because the default `HIPFORT` installation only supports gfortran,  we use a custom installation  prepared in the summer school project. This package provide Fortran modules compatible with the Cray Fortran compiler as well as direct use of hipfort with the Fortran Cray Compiler wrapper (ftn). 

The package was installed via:
```bash
git clone https://github.com/ROCmSoftwarePlatform/hipfort.git
cd hipfort;
mkdir build;
cd build;
cmake -DHIPFORT_INSTALL_DIR=<path-to>/HIPFORT -DHIPFORT_COMPILER_FLAGS="-ffree -eZ" -DHIPFORT_COMPILER=ftn -DHIPFORT_AR=${CRAY_BINUTILS_BIN_X86_64}/ar -DHIPFORT_RANLIB=${CRAY_BINUTILS_BIN_X86_64}/ranlib  ..
make -j 64 
make install
```

We will use the Cray 'ftn' compiler wrapper as you would do to compile any fortran code plus some additional flags:
```bash
export HIPFORT_HOME=/project/project_465000536/appl/HIPFORT
ftn -I$HIPFORT_HOME/include/hipfort/amdgcn "-DHIPFORT_ARCH=\"amd\"" -L$HIPFORT_HOME/lib -lhipfort-amdgcn $LIB_FLAGS -c <fortran_code>.f90 
CC -xhip -c <hip_kernels>.cpp
ftn  -I$HIPFORT_HOME/include/hipfort/amdgcn "-DHIPFORT_ARCH=\"amd\"" -L$HIPFORT_HOME/lib -lhipfort-amdgcn $LIB_FLAGS -o main <fortran_code>.o hip_kernels.o
```
This option gives enough flexibility for calling HIP libraries from Fortran or for a mix of OpenMP/OpenACC offloading to GPUs and HIP kernels/libraries.

## Running in LUMI

### Pure MPI

Programs need to be executed via the batch job system. Simple job running with 4 MPI tasks can be submitted with the following batch job script:
```
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_465000536
#SBATCH --partition=standard
#SBATCH --reservation=summerschool_standard
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1  

srun my_mpi_exe
```

Save the script *e.g.* as `job.sh` and submit it with `sbatch job.sh`.
The output of job will be in file `slurm-xxxxx.out`. You can check the status of your jobs with `squeue -u $USER` and kill possible hanging applications with
`scancel JOBID`.

The reservation `summerschool` is available during the course days and it
is accessible only with the training user accounts.

### Pure OpenMP

For pure OpenMP programs one should use only one node and one MPI task per nodesingle tasks and specify the number of cores reserved
for threading with `--cpus-per-task`:
```
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_465000536
#SBATCH --partition=standard
#SBATCH --reservation=summerschool_standard
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

# Set the number of threads based on --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
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
#SBATCH --reservation=summerschool_standard
#SBATCH --time=00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=4
# Set the number of threads based on --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun my_exe
```


### GPU programs

When running GPU programs, few changes need to made to the batch job
script. The `partition` is are now different, and one must also request explicitly given number of GPUs per node with the
`--gpus-per-node=8` option. As an example, in order to use a
single GPU with single MPI task and a single thread use:
```
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_465000536
#SBATCH --partition=standard-g
#SBATCH --reservation=summerschool_standard-g
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

## Debugging

See [the MPI debugging exercise](mpi/debugging),
[CSC user guide](https://docs.csc.fi/computing/debugging/), and
[LUMI documentation](https://docs.lumi-supercomputer.eu/development/)
for possible debugging options.

## Performance analysis with TAU and Omniperf

`TAU` and `Omniperf` can be used to do performance analysis. 
In order to use TAU one only has to load the modules needed to run the application be ran and set the paths to the TAU install folder:
```
export TAU=/project/project_465000536/appl/tau/2.32/craycnl
export PATH=$TAU/bin:$PATH
```
In order to use omniperf load the follwoing modules:
```
module use /project/project_465000536/Omni/omniperf/modulefiles
module load omniperf
module load cray-python
``` 
More information about TAU can be found in [TAU User Documentation](https://www.cs.uoregon.edu/research/tau/docs/newguide/), while for Omniperf at [Omniperf User Documentation](https://github.com/AMDResearch/omniperf)
