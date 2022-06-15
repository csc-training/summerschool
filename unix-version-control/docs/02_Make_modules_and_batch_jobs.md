---
title:  Make, modules and batch jobs
event:  CSC Summer School in High-Performance Computing 2022
lang:   en
---

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
- Run make with: `make target`
    * Without an argument, make runs the first rule in the file
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


# Modules and batch job system {.section}


# Module environment

- Supercomputers have a large number of users with different needs for
  development environments and applications
- _Environment modules_ provide a convenient way to dynamically change the
  user's environment
- In this way, different compiler suites and application versions can be used
  more easily
    * They basically just change where things are "pointing". So when you run
      `gcc` the loaded module decides whether you are using version 4.9 or 5.3
      or 6.0 and so on
    * Most programs requires loading a module to be accessible


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


# Example batch job script

```bash
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=yourproject
#SBATCH –-partition=test
#SBATCH –-time=00:10:00
#SBATCH –-ntasks=80
#SBATCH --mem-per-cpu=4000

srun myprog
```

- More examples:
    - https://docs.csc.fi/computing/running/example-job-scripts-puhti/
    - https://docs.csc.fi/computing/running/example-job-scripts-mahti/


# Running batch jobs

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
- Show job resource usage with `sacct`:
  ```bash
  sacct jobid
  ```
