---
title:  Parallel debugging
author: CSC Training
date:   2021
lang:   en
---

# Parallel debugging {.section}

# Debugging

- Bugs are evident in any non-trivial program
- Crashes (Segmentation fault) or incorrect results
- Parallel programs can have also deadlocks and race conditions

# Finding bugs

- Print statements in the code
    - Typically cumbersome, especially with compiled languages
	- Might result in lots of clutter in parallel programs
	- Order of printouts from different processes is arbitrary
- "Standard" debuggers
    - `gdb`: common command line debugger
    - Debuggers within IDEs, e.g. VS Code
    - No proper support for parallel debugging
- Parallel debuggers
    - **Allinea DDT**, Totalview (commercial products)
	
# Common features in debuggers

- Setting breakpoints and watchpoints
- Executing code line by line
- Stepping into / out from functions
- Investigating values of variables
- Investigating program stack
- Parallel debuggers allow all of the above on per process/thread
  basis

# Web resources

- Defensive programming and debugging online course <https://www.futurelearn.com/courses/defensive-programming-and-debugging>
- Using `gdb` for parallel debugging <https://www.open-mpi.org/faq/?category=debugging>
- Memory debugging with Valgrind <https://valgrind.org/docs/manual/mc-manual.html#mc-manual.mpiwrap>

# Demo: using Allinea DDT {.section}

# Using Allinea DDT

- Code needs to be compiled with debugging option `-g`
- Compiler optimizations might complicate debugging (dead code
  elimination, loop transformations, *etc.*), recommended to
  compile without optimizations with `-O0`
    - Sometimes bugs show up only with optimizations
- In CSC environment DDT is available via `module load ddt`
- Debugger needs to be started in an interactive session
```bash
module load ddt
export SLURM_OVERLAP=1
salloc --nodes=1 --ntasks-per-node=2 --account=project_xxx -p small
ddt srun ./buggy
```
- NoMachine remote desktop is recommended for smoother GUI performance

