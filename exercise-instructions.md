## General exercise instructions

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

### Computing servers

Exercises can be carried out using the CSC's Puhti supercomputer. See [CSC User Documentation](https://docs.csc.fi/support/tutorials/puhti_quick/) 
for general instructions on using Puhti.

In case you have working parallel program development environment in your
laptop (Fortran or C compiler, MPI development library, etc.) you may also use
that. Note, however, that no support for installing MPI environment can be provided during the course.

Puhti and Mahti can be accessed via ssh using the
provided username (`trainingxxx`) and password:
```
ssh -Y training000@puhti.csc.fi
```
or
```
ssh -Y training000@mahti.csc.fi
```

For easier connecting we recommend that you set uo *ssh keys* along the instructions in 
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


## Compilation and running

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

### HDF5

In order to use HDF5 in Puhti, you need the load the HDF5 module with MPI I/O support:
```
module load hdf5/1.10.4-mpi
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

#### Running in Puhti

In Puhti, programs need to be executed via the batch job system. Simple job running with 4 MPI tasks can be submitted with the following batch job script:
```
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_2000745
#SBATCH --partition=small
#SBATCH --reservation=advance-mpi
#SBATCH --time=00:05:00
#SBATCH --ntasks=4

srun my_mpi_exe
```

Save the script *e.g.* as `job.sh` and submit it with `sbatch job.sh`. 
The output of job will be in file `slurm-xxxxx.out`. You can check the status of your jobs with `squeue -u $USER` and kill possible hanging applications with
`scancel JOBID`.

The reservation `mpi_intro` is available during the course days and it
is accessible only with the training user accounts.

#### Running in local workstation

In most MPI implementations parallel program can be started with the `mpiexec` launcher:
```
mpiexec -n 4 ./my_mpi_exe
```

### Debugging

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

For smoother GUI performance, we recommend using [NoMachine remote
desktop](https://docs.csc.fi/support/tutorials/nomachine-usage/) to
connect to Puhti.

### Performance analysis with ScoreP / Scalasca

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
the experiment directory (e.g. `scorep_my_mpi_app_8_trace`). For smoother GUI
performance, we recommend using [NoMachine remote desktop](https://docs.csc.fi/support/tutorials/nomachine-usage/) 
to connect to Puhti.

More information about Scalasca can be found in [CSC User Documentation](https://docs.csc.fi/apps/scalasca/)

