---
title:  Working in supercomputers
event:  CSC Summer School in High-Performance Computing 2025
lang:   en
---

# Outline

- Connecting to LUMI and CSC supercomputers
- File system in LUMI and CSC supercomputers
- Module system
- Building applications
- Running applications via batch job system

# Connecting to LUMI and CSC supercomputers {.section}

# Anatomy of a supercomputer

<!-- Copyright CSC -->
![](img/cluster_diagram.svg){.center width=100%}

# Connecting to LUMI and CSC supercomputers

- SSH with public key authentication is used to connect the login node
  - <https://github.com/csc-training/summerschool/wiki/Setting-up-CSC-account-and-SSH>
- Also web interfaces exist
  - <https://www.lumi.csc.fi>
  - <https://www.mahti.csc.fi>
  - <https://www.puhti.csc.fi>

# File system in LUMI and CSC supercomputers {.section}

# Directory structure

- LUMI and CSC supercomputers have separate file systems
  - Files need to be explicitly copied between LUMI, Mahti, and Puhti
- Directory structure is common in all systems

|            |Owner   |Environment variable|Path                 |
|------------|--------|--------------------|---------------------|
|**home**    |Personal|`$HOME`             |`/users/<user-name>` |
|**projappl**|Project |Not available       |`/projappl/<project>`|
|**scratch** |Project |Not available       |`/scratch/<project>` |

- See `lumi-workspaces` on LUMI or `csc-workspaces` on Mahti and Puhti

# Using project-level storage space

- Common practice: create your personal directory under scratch:
  ```bash
  mkdir -p /scratch/<project>/$USER
  cd /scratch/<project>/$USER
  ```
- Use this personal work space during summer school to avoid file conflicts with other project members

# Parallel distributed file system: Lustre

- All storage is accessed via interconnect and it is a *shared resource* among *all users* in the supercomputer<br>
  - One should avoid putting unnecessary load to the file system to keep the system responsive
- LUMI and CSC supercomputers use Lustre as the parallel distributed file system<br>
  - Lustre is designed for efficient parallel I/O for large files
  - Excessively accessing file metadata or opening and closing files puts stress on *metadata servers* and can cause slowness in the file system (especially if *all users* do so)

# Being nice to Lustre (and other users)

- Avoid accessing a large number of small files
  - Practical example: Python environments are typically containerized in supercomputers to avoid a significant performance hit due to accessing thousands of small files when loading the enviroment
- Avoid `ls -l` and use plain `ls` instead if you don't need the extra metadata
  - Less stress on the metadata servers
- Use Lustre tools (e.g., `lfs find`) instead of regular file system tools (e.g. `find`)
  - Less stress on the metadata servers


# Module system {.section}

# Module environment

- Supercomputers have a large number of users with different needs for
  development environments and applications
- _Environment modules_ provide a convenient way to dynamically change the
  user's environment
- With modules different compiler suites and application versions can be used smoothly
  - Changing compiler module loads automatically also correct versions of libraries
  - Loading a module for application sets up the correct environment with a single command


# Common module commands

<div class="column">
`module load mod`
  : Load module **mod** in shell environment

`module unload mod`
  : Remove module **mod** from environment

`module list`
  : List loaded modules

`module avail`
  : List all available modules
</div>

<div class="column">
`module spider mod`
  : Search for module **mod**

`module show mod`
  : Get information about module **mod**

`module switch mod1 mod2`
  : Switch loaded **mod1** to **mod2**
</div>


# Building applications {.section}

# Compiling and linking

<div class=column>
- A compiler turns a source code file into an object file that contains
  machine code that can be executed by the processor
- A linker combines several compiled object files into a single executable file
- Together, compiling and linking is called building
</div>
<div class=column>
![](img/building.svg){.center}
</div>

# Compiling and linking

Single file source code:

```bash
cc main.c -o main
```

- In practice programs are separated into several files
  <br>$\Rightarrow$ complicated dependency structures
- Building large programs takes time
  - Could we just rebuild the parts that changed?
- Having different options when building
  - Debug versions, enabling/disabling features, etc.


# Make

- Make allows you to define how to build individual parts of a program
  and how they depend on each other. This enables:
  - Building parts of a program and only rebuilding necessary parts
  - Building different version and configurations of a program

![](img/depend.png){.center width=40%}

# Make rules

<div class=column>
- Make **rules** define how some part of your program is built
    - **Target**: the output file (or aim) of your rule
    - **Dependency**: which other targets your target depends on
    - **Recipe**: how you produce your target
- Rules are defined in a file which is by default called `Makefile`
</div>

<div class=column>
_A make rule_
![](img/rule.png){.center width=100%}
</div>


# Example Makefile

<div class=column>
- Dependencies can be files or other targets
- Recipes consist of one or more shell commands
    - Recipe lines start with a **tabulator**
- If the dependencies are newer than the target, make runs the recipe
- Run first rule: `make`
- Run specific rule: `make <target>`
</div>

<div class=column>
```makefile
main: main.c functions.o
    gcc -c main.c
    gcc -o main main.o functions.o

functions.o: functions.c
    gcc -c functions.c

.PHONY: clean
clean:
    rm -f main.o functions.o main
```
</div>

# Variables and patterns in rules

<div class=column>
- It is possible to define variables, for example compiler and link commands
  and flags
- Targets, dependencies and recipes can contain special wild cards
- Rerunning all the recipes can be forced with `make -B`
</div>

<div class=column>
```makefile
CC=cc
CCFLAGS=-O3

# Files of the form filename.o depend on
# filename.c
%.o: %.c
        $(CC) $(CCFLAGS) -c $< -o $@
```
</div>

# Build generators

- In a large software projects, figuring out all the dependencies between
  software modules can be difficult
- `Makefile` is not necessarily portable
- In order to improve portability and make dependency handling easier, build generators
  are often used
    - Select automatically correct compilers and compiler options
    - Create `Makefile` from simpler template
- **GNU Autotools** and **cmake** are the most common build generators in HPC


# Running applications via batch job system {.section}

# Batch queue system

- On a cluster, instead of running a program instantly, you submit your
  program/simulation (aka job) to a queue and the system will then execute it
  once the resources are available
    - The queue enables effective and fair resource usage
    - CSC uses SLURM as the queue system

- When running a job on a supercomputer you need to:
    - Describe how you want to run the job and what resources you need
    - Add a command that launches your program
    - Submit your job to a queue


# Example SLURM batch job script

```bash
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=<project_id>
#SBATCH --partition=small
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128

# srun launches "nodes * ntasks-per-node" MPI tasks of myprog
srun ./myprog
```

- More examples:
  - <https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/batch-job/>
  - <https://docs.csc.fi/computing/running/example-job-scripts-mahti/>
  - <https://docs.csc.fi/computing/running/example-job-scripts-puhti/>


# Submitting batch jobs

- Submit your batch job script to the queue using `sbatch`:
  ```bash
  sbatch job.sh
  ```
- You can override parameters in the job script from commandline:
  ```bash
  sbatch --nodes=1 --ntasks-per-node=4 --partition=debug job.sh
  ```

# Managing batch jobs

- You can follow the status of your jobs with `squeue`:
  ```bash
  squeue --me
  ```
- If something goes wrong, you can cancel your job with `scancel`:
  ```bash
  scancel jobid
  ```
  (here the jobid is the numeric ID of the job)
- Show job resource usage (for completed jobs) with `sacct`:
  ```bash
  sacct jobid
  ```

# Running interactive jobs

- Alternatively to `sbatch`, you can submit a job to the queue using `srun`
  ```bash
  srun --account=... --partition=small --nodes=2 --ntasks-per-nodes=128 --time=00:10:00 ./myprog
  ```
  In this case the output will be shown on the terminal  (job will fail if the connection is lost).

- Yet another way is to create an allocation with `salloc`:
  ```bash
  salloc --account=... --partition=small --nodes=2 --ntasks-per-nodes=128 --time=00:30:00
  ```
  Once the allocation is made, the `srun` command will execute in that allocation:
  ```bash
  srun --ntasks=32 --cpus-per-task=8 ./myprog
  ```

# Useful environment variables

Following variables are available inside Slurm scripts:

- `SLURM_JOBID`: jobid
- `SLURM_JOB_NAME` : name given in `job_name`
- `SLURM_JOB_NODELIST` : list of nodes the job will run

Following variables are available inside program launched by `srun`:

- `SLURM_PROCID` : global id of process
- `SLURM_LOCALID` : node local id of process

# Useful environment variables

Following variables override the corresponding variables in the batch script and command line:

- `SBATCH_ACCOUNT` : `--account`
- `SBATCH_PARTITION` : `--partition`

See `man sbatch` for full list.


#  HPC filesystems and Lustre {.section}

# Filesystems on CSC supercomputers

The filesystem used on CSC systems (Puhti, Mahti, Lumi) is called **Lustre**.

- Parallel: data is distributed across many storage drives
- Files can be accessed from all tasks (user permissions still apply)
- Lustre is very common in HPC in general, not just at CSC

Many systems also provide node-local disk area for temporary storage

- `/tmp`, `$TMPDIR`, `$LOCAL_SCRATCH` *etc.* depending on the system
- Sometimes the temporary storage may reside directly in memory (`/tmp` on Lumi compute nodes). Consult system docs

# Lustre architecture

<div class="column">

- Files are chunked up and spread across multiple **storage servers** as **objects**
- Dedicated **metadata server(s)** (MDS): file names, owners, permissions, ...
- **Client**: HPC nodes that access the data

</div>

<div class="column">

![](img/lustre-architecture.svg)
Clients interact with MDS once to gain OST access, then I/O to objects directly.

- Allows for **very high, parallel I/O bandwidth!**

</div>

# Being nice to Lustre as an HPC user

Every file lookup, file creation/deletion, permission change *etc.* is processed by the metadata server.

- Metadata servers are shared by everyone using the supercomputer!
- Commands like `ls` unresponsive? Servers may be under heavy load

**Please be mindful of other users!**

- Lustre best practices in docs: <https://docs.csc.fi/computing/lustre/#best-practices>


# Some Lustre best practices

- Avoid creating/accessing small files in large quantities (*eg.* in scripts)
- Avoid listing extended attributes (timestamps *etc.*) when not necessary
  - Prefer `ls` over `ls -l` for file listing
  - `LUE` tool for size queries: <https://docs.csc.fi/support/tutorials/lue/>
- Consider using local disk when compiling code: `$TMPDIR` or similar non-Lustre location
- Be careful when installing Python packages with `Conda` or `pip`!
  - These have the "many small files" problem, especially with dependencies
  - See <https://docs.lumi-supercomputer.eu/software/installing/python/>

# Summary {.section}

# Summary

- Login nodes are entry points to a supercomputer
- Calculations are submitted as jobs to the queueing system
- Modules and build tools help managing environment and software
