---
title:  Working in supercomputers
event:  CSC Summer School in High-Performance Computing 2024
lang:   en
---

# Outline

- Connecting to supercomputers
- Directory structure in CSC supercomputers
- Module system
- Building applications with `make`
- Running applications via batch job system

<br>
Some information provided here is specific to LUMI, Mahti, and Puhti, some more general

# Anatomy of supercomputer
 ![](images/cluster_diagram.jpeg){.center width=80%}
 
# Connecting to supercomputers

- `ssh` is used to connect to all CSC supercomputers
```
ssh <my_user_id>@lumi.csc.fi
```
- With **ssh keys** one can login without password
    - Automatically setup for LUMI
    - In Mahti and Puhti need to create and setup manually

```
ssh-keygen -o -a 100 -t ed25519 # Not needed if key exists
ssh-copy-id <my_user_id>@mahti.csc.fi
```


# Directory structure in CSC supercomputers

- All the CSC supercomputers have separate file systems
    - Files need to be explicitly copied between LUMI, Mahti and Puhti
- All the CSC supercomputers share a commong directory structure

|            |Owner   |Environment variable|Path                 |
|------------|--------|--------------------|---------------------|
|**home**    |Personal|`${HOME}`           |`/users/<user-name>` |
|**projappl**|Project |Not available       |`/projappl/<project>`|
|**scratch** |Project |Not available       |`/scratch/<project>` |

# Modules {.section}

# Module environment

- Supercomputers have a large number of users with different needs for
  development environments and applications
- _Environment modules_ provide a convenient way to dynamically change the
  user's environment
- In this way, different compiler suites and application versions can be used
  more easily
    * Changing compiler module loads automatically also correct versions of libraries 
    * Loading a module for application sets up the correct environment with single 
      command


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


# Make {.section}

# Compiling and linking

<div class=column>
- A compiler turns a source code file into an object file that contains
  machine code that can be executed by the processor
- A linker combines several compiled object files into a single executable
  file
- Together, compiling and linking can be called building
</div>
<div class=column>
![](images/building.svg){.center}
</div>

# Compiling and linking

Single file source code:

```bash
cc main.c -o main
```
 
<div class=column>
 
<small>
 In practice programs be separated into several files
  <br>$\Rightarrow$ complicated dependency structures
- Building large programs takes time
    - could we just rebuild the parts that changed?
- Having different options when building
    - debug versions, enabling/disabling features, etc.
</small>

</div>

<div class=column>
![](images/depend.png){.center width=40%}
</div>

# Compiling and linking: possible problems

- Programs should usually be separated into several files
  <br>$\Rightarrow$ complicated dependency structures
- Building large programs takes time
    - could we just rebuild the parts that changed?
- Having different options when building
    - debug versions, enabling/disabling features, etc.

# Make

- Make allows you to define how to build individual parts of a program
  and how they depend on each other. This enables:
    - Building parts of a program and only rebuilding necessary parts
    - Building different version and configurations of a program

![](images/depend.png){.center width=40%}

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
![](images/rule.png){.center width=100%}
</div>


# Simple Makefile example

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
- Rerunning all the recipies can be forced with `make -B`
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


# Batch queue system {.section}

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
- This is done with a batch job script


# Example SLURM batch job script

```bash
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=<project_id>
#SBATCH –-partition=small
#SBATCH –-time=00:10:00
#SBATCH –-nodes=2
#SBATCH –-ntasks-per-nodes=128

# srun launches "nodes * ntasks-per-nodes" copies of myprog
srun myprog
```

- More examples:
    - <https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/batch-job/>
    - <https://docs.csc.fi/computing/running/example-job-scripts-puhti/>
    - <https://docs.csc.fi/computing/running/example-job-scripts-mahti/>


# Running batch jobs under SLURM

- Submit your batch job script to the queue using `sbatch`
  ```bash
  sbatch my_job.sh
  ```
- You can follow the status of your jobs with `squeue`:
  ```bash
  squeue -u my_username
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
  
# Interactive jobs

Alternatively to `sbatch` one can submit a job to the queue using `srun`

```bash
srun --account=<...> –-partition=<...> –-time=00:10:00 –-nodes=2 –-ntasks-per-nodes=128 ./myprog
```

In this case the output will be shown on the terminal  (job will fail if the connection is lost). 

When debugging or doing performance analysys the user needs to interact with application on the compute nodes.

```bash
salloc --account=<project_id> –-partition=small –-nodes=2 –-ntasks-per-nodes=128 --time=00:30:00
```
Once the allocation is made, this command will start a shell on the login node.

```bash
srun --ntasks=32 --cpus-per-task=8 ./my_interactive_prog
```

# Useful Slurm environment variables


Following variables are available inside Slurm scripts:

<small>

- `SLURM_JOB_NAME` : name given in `job_name`
- `SLURM_JOBID`
- `SLURM_JOB_NODELIST` : list of nodes the job will run

</small>

Following variables are available inside program launched by `srun`

<small>

- `SLURM_PROCID` : global id of process 
- `SLURM_LOCALID` : node local id of process 

</small>

Following variables override the corresponding variables in the batch script

<small>

- `SBATCH_ACCOUNT` : `--account`
- `SBATCH_PARTITION` : `--partition`

</small>

