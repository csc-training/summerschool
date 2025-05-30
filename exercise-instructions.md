# General exercise instructions

## Working with the summer school repository

The repository of the summer school can be cloned via:
```
git clone https://github.com/csc-training/summerschool.git
```

#### Repository structure

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
exercises. Note, however, that no support for installing MPI environment or ROCM can be
provided during the course. Otherwise, you can use CSC supercomputers for
carrying out the exercises.

## Using supercomputers

Exercises can be carried out using the [LUMI](https://docs.lumi-supercomputer.eu/) supercomputer.

LUMI can be accessed via ssh using the provided username and ssh key pair:
```
ssh -i <path-to-private-key> <username>@lumi.csc.fi
```

#### Working with git (OPTIONAL)

In order to push code to your own fork, you need to **add your SSH public key in LUMI** (created on LUMI, **not** the used to log into LUMI)
to your github account. The SSH key can be added to your github account using a browser. In your github profile go to "Settings"->"SSH and GPG keys"->"New SSH key" and copy-paste output of the following command:
```
cat $HOME/.ssh/id_rsa.pub
```

Once succesful, make sure you are in your personal workspace in the **scratch** area `/scratch/project_465000536/$USER`, clone the repository, and create a branch:
```
git clone git@github.com:<my-github-id>/summerschool.git
git checkout -b hpcss25
```

If you haven't used git before in LUMI, you need to add also your identity:
```
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
```

The default editor for commit messages is *vim*, if you prefer something else you can add
to the file `$HOME/.bashrc` *e.g.*
```
export EDITOR=nano
```

#### Disk areas

All the exercises in the supercomputers should be carried out in the
**scratch** disk area. The name of the scratch directory can be
queried with the command `lumi-workspaces`. As the base directory is
shared between members of the project, you should create your own
directory:
```
mkdir -p /scratch/project_465001194/$USER
cd /scratch/project_465001194/$USER
```

#### Editors

For editing program source files you can use e.g. the *nano* editor:

```
nano prog.f90
```
(`^` in nano's shortcuts refer to **Ctrl** key, *i.e.* in order to save the file and exit the editor press `Ctrl+X`)
Also other popular editors such as *emacs* and *vim* are available.

## Compilation

LUMI has several programming environments. For the summer school, we recommend that you use the Cray tools.

For CPU programming use:
```
module load PrgEnv-cray/8.4.0
module load LUMI/23.09
module load partition/C
```
For GPU programming use:
```
module load PrgEnv-cray/8.4.0
module load LUMI/23.09
module load partition/G
module load rocm/5.4.6
```

#### MPI

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

#### OpenMP (threading with CPUs)

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

#### HDF5

In order to use HDF5 in CSC supercomputers, you need the load the HDF5 module with MPI I/O support.
The appropriate module in **Lumi** is
```
module load cray-hdf5-parallel
```

No special flags are needed for compiling and linking, the compiler wrappers take care of them automatically.

Usage in your local workstation may vary.

#### OpenMP offloading

On **Lumi**, the following modules are required:

```bash
module load PrgEnv-cray/8.4.0
module load LUMI/23.09
module load partition/G
module load rocm/5.4.6
```

On **Lumi**, to compile your program, use
```bash
CC -fopenmp <source.cpp>
```
**NOTE!** The `-fopenmp` option behaves differently depending on which module are loaded. If `partition/L` or `partition/C` is loaded it will use compiling options for creating code for multi-core cpus. If `partition/G` is loaded it will use compiling options to create code for offloading on GPUs.

#### HIP

Use the following modules :

```bash
module load PrgEnv-cray/8.4.0
module load LUMI/23.09
module load partition/G
module load rocm/5.4.6
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
#### HIPFORT
The following modules are required:
```bash
module load PrgEnv-cray/8.4.0
module load LUMI/23.09
module load partition/G
module load rocm/5.4.6
```

Because the default `HIPFORT` installation only supports `gfortran`, we use a custom installation prepared in the summer school project. This package provide Fortran modules compatible with the Cray Fortran compiler as well as direct use of `hipfort` with the Fortran Cray Compiler wrapper (`ftn`).

The package was installed via:
```bash
# In some temporary folder
wget https://github.com/ROCm/hipfort/archive/refs/tags/rocm-6.1.0.tar.gz # one can try various realeases
tar -xvzf rocm-6.1.0.tar.gz;
cd hipfort-rocm-6.1.0;
mkdir build;
cd build;
cmake -DHIPFORT_INSTALL_DIR=<path-to>/HIPFORT -DHIPFORT_COMPILER_FLAGS="-ffree -eZ" -DHIPFORT_COMPILER=<path-to>/ftn -DHIPFORT_AR=${CRAY_BINUTILS_BIN_X86_64}/ar -DHIPFORT_RANLIB=${CRAY_BINUTILS_BIN_X86_64}/ranlib  ..
make -j 64
make install
```
Where `<path-to>/ftn` can be obtain by running `which ftn`.

We will use the Cray 'ftn' compiler wrapper as you would do to compile any fortran code plus some additional flags:
```bash
export HIPFORT_HOME=/projappl/project_465001194/apps/HIPFORT
ftn -I$HIPFORT_HOME/include/hipfort/amdgcn "-DHIPFORT_ARCH=\"amd\"" -L$HIPFORT_HOME/lib -lhipfort-amdgcn $LIB_FLAGS -c <fortran_code>.f90
CC -xhip -c <hip_kernels>.cpp
ftn  -I$HIPFORT_HOME/include/hipfort/amdgcn "-DHIPFORT_ARCH=\"amd\"" -L$HIPFORT_HOME/lib -lhipfort-amdgcn $LIB_FLAGS -o main <fortran_code>.o hip_kernels.o
```
This option gives enough flexibility for calling HIP libraries from Fortran or for a mix of OpenMP/OpenACC offloading to GPUs and HIP kernels/libraries.

### SYCL
#### OneAPI + AMD Plug-in
Set-up the modules and paths:
```
module load LUMI/24.03
module load partition/G
module load rocm/6.2.2
source /projappl/project_462000956/apps/intel/oneapi/setvars.sh --include-intel-llvm
export  HSA_XNACK=1 # enables managed memory
export MPICH_GPU_SUPPORT_ENABLED=1                                # Needed for using GPU-aware MPI
```
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
#SBATCH --account=project_465001194
#SBATCH --partition=small
#SBATCH --reservation=CSC_summer_school_cpu
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1

srun my_mpi_exe
```

Save the script *e.g.* as `job.sh` and submit it with `sbatch job.sh`.
The output of the job will be in the file `slurm-xxxxx.out`. You can check the status of your jobs with `squeue -u $USER` and kill possible hanging applications with `scancel JOBID`.

The reservation `summerschool` is available during the course days and it is accessible only with the training user accounts.

The same result can be achieved using directly `srun`
```
srun --job-name=example --account=project_465001194 --partition=small --reservation=CSC_summer_school_cpu --time=00:05:00 --nodes=1 --ntasks-per-node=4 --cpus-per-task=1  my_mpi_exe
```
#### Pure OpenMP

For pure OpenMP programs one should use only one node and one MPI task per node and specify the number of cores reserved
for threading with `--cpus-per-task`:
```
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_465001194
#SBATCH --partition=small
#SBATCH --reservation=CSC_summer_school_cpu
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

# Set the number of threads based on --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun my_omp_exe
```

The same result can be achieved using directly `srun`
```
srun --job-name=example --account=project_465001194 --partition=small --reservation=CSC_summer_school_cpu --time=00:05:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=4   my_omp_exe
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
#SBATCH --account=project_465001194
#SBATCH --partition=small
#SBATCH --reservation=CSC_summer_school_cpu
#SBATCH --time=00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=4
# Set the number of threads based on --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun my_exe
```

The same result can be achieved using directly `srun`
```
srun --job-name=example --account=project_465001194 --partition=small --reservation=CSC_summer_school_cpu --time=00:05:00 --nodes=2 --ntasks-per-node=32 --cpus-per-task=4   my_omp_exe
```
#### GPU programs

When running GPU programs, few changes need to made to the batch job
script. The `partition` is now different, and one must also request explicitly a given number of GPUs per node with the
`--gpus-per-node=X` option. As an example, in order to use a
single GPU with single MPI task and a single thread use:
```
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_465001194
#SBATCH --partition=small-g
#SBATCH --reservation=CSC_summer_school_gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00

srun my_gpu_exe
```

The same result can be achieved using directly `srun`
```
srun --job-name=example --account=project_465001194 --partition=small-g --reservation=CSC_summer_school_gpu --time=00:05:00 --gpus-per-node=1 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1  my_gpu_exe
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

## Performance analysis with TAU and Omniperf  (**OUTDATED, check the [application-performance](/application-performance) folder!**)
```
# Create installation directory
mkdir -p .../appl/tau
cd .../appl/tau

# Download TAU
wget https://www.cs.uoregon.edu/research/tau/tau_releases/tau-2.32.tar.gz
tar xvf tau-2.32.tar.gz
mv tau-2.32 2.32

# Go to TAU directory
cd 2.32

./configure -bfd=download -otf=download -unwind=download -dwarf=download -iowrapper -cc=cc -c++=CC -fortran=ftn -pthread -mpi -phiprof -papi=/opt/cray/pe/papi/6.0.0.15/
make -j 64

./configure -bfd=download -otf=download -unwind=download -dwarf=download -iowrapper -cc=cc -c++=CC -fortran=ftn -pthread -mpi -papi=/opt/cray/pe/papi/6.0.0.15/ -rocm=<path-to>/rocm/x.y.z/ -rocprofiler=<path-to>/rocm/x.y.z/rocprofiler
make -j 64

./configure -bfd=download -otf=download -unwind=download -dwarf=download -iowrapper -cc=cc -c++=CC -fortran=ftn -pthread -mpi -papi=/opt/cray/pe/papi/6.0.0.15/ -rocm=<path-to>/rocm/x.y.z/ -roctracer=<path-to>/rocm/x.y.z/roctracer
make -j 64
```

`TAU` and `Omniperf` can be used to do performance analysis.
In order to use TAU one only has to load the modules needed to run the application be ran and set the paths to the TAU install folder:
```
export TAU=<path-to>/tau/2.32/craycnl
export PATH=$TAU/bin:$PATH
```
Profiling mpi code:
```
srun --cpus-per-task=1 --account=project_465001194 --nodes=1 --ntasks-per-node=4 --partition=standard --time=00:05:00 --reservation=summerschool_standard tau_exec -ebs ./mandelbrot
```
In order to to see the `paraprof` in browser use `vnc`:
```
module load lumi-vnc
start-vnc
```
Visualize:
```
paraprof
```
This will open the application in a vnc window.

Alternatively one can use the [Open on Demand](https://www.lumi.csc.fi/) interface. This way one can have access to a desktop running on LUMI.

Tracing:

```
export TAU_TRACE=1
srun --cpus-per-task=1 --account=project_465001194 --nodes=1 --ntasks-per-node=4 --partition=standard --time=00:05:00 --reservation=summerschool_standard tau_exec -ebs ./mandelbrot
tau_treemerge.pl
tau_trace2json tau.trc tau.edf -chrome -ignoreatomic -o app.json
```

Copy `app.json`  to local computer, open ui.perfett.dev and then load the `app.json` file.
## Omniperf
Installing Omniperf is straightforward follwing the instructions from the [official webpage](https://amdresearch.github.io/omniperf/installation.html#client-side-installation)
In order to use omniperf load the following modules:
```
module use <path-ro>/Omni/omniperf/modulefiles
module load omniperf
module load cray-python
srun -p standard-g --gpus 1 -N 1 -n 1 -c 1 --time=00:30:00 --account=project_465001194 omniperf profile -n workload_xy --roof-only --kernel-names  -- ./heat_hip
omniperf analyze -p workloads/workload_xy/mi200/ > analyse_xy.txt
```
In additition to this one has to load the usual modules for running GPUs. Keep in mind the the above installation was done with `rocm/5.4.6`.
It is useful add to the compilation of the application to be analysed the follwing `-g -gdwarf-4`.

More information about TAU can be found in [TAU User Documentation](https://www.cs.uoregon.edu/research/tau/docs/newguide/), while for Omniperf at [Omniperf User Documentation](https://github.com/AMDResearch/omniperf)
