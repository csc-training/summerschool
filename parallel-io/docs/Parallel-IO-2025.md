---
title:  Input/Output (I/O) in HPC
event:  CSC Summer School in High-Performance Computing 2025
lang:   en
---

# PLAN

- Remove section about HDF5 altogether? Reasoning: can't teach the API usage in short amount of time
- Come up with an exercise that has the HDF5 boilerplate already done.
- Demonstrate reading of HDF5 files. Command line tools, Python

# Common I/O use cases in HPC

- Store simulation results on disk
- Store full state of the simulation, eg. for visualization or as a checkpoint for continuing later
- Read configuration files, input parameters, earlier checkpoint files...

Example: Climate simulation code `ICON` produces ~2TB of data per simulated month.

- 30-year model: 700TB at a rate of 100MB/s
- **I/O can easily become a bottleneck if not properly planned for!**

<!-- Numbers obtained from JE. They correspond to a "recent" (2025) simulation for `ClimateDT` using 5km resolution. -->

# Implementing good I/O: Considerations

- Performance, scalability, reliability
- Ease of use of output (number of files, format)
- Portability
- Code maintainability and ease of use

Often it is not possible or practical to achieve all of the above - one needs to prioritize case-by-case.

Supercomputers use parallel file systems that have their own quirks and performance implications.

# A comment on terminology...

You are probably familiar with so-called "standard library" I/O routines

- **C**: `fread`, `fwrite` etc. from `stdio.h`
- **fortran**: TODO

These are usually implemented through operating system `syscalls`;\
`POSIX`-routines in most Linux distros.

- Most programs should ***never*** need to call POSIX directly! Use standard libs or dedicated I/O libs instead
- HPC texts sometimes use the term "POSIX I/O" analogously with any non-parallel I/O library. Try not to get confused :)


# Common I/O strategies in HPC {.section}

# Approaches to I/O in a parallel program

Small programs _can_ manage with standard I/O routines:

- Do all I/O from one process, using MPI to gather/scatter the data (**"spokesman"**)
- Use separate I/O file(s) for each process (**"file-per-process"**)

Large programs often need genuine *parallel I/O*:

- Many processes read/write to same file(s) in a coordinated, preferably asynchronous fashion
- Standards and libraries: **MPI-IO**, **HDF5**, **NetCDF**, ...

# Spokesman I/O strategy

<div class="column">
- One process takes care of all I/O using standard, serial routines (`fprintf`, `fwrite`, ...)
- Usually requires a lot of MPI communication
- Can be a good option for small files (eg. input
    files)
- Does not scale, single writer is a bottleneck!
</div>
<div class="column">
![](img/posix-spokesman.png)
</div>

Exercise: `spokesman`

# File-per-process I/O

<div class="column">
- Each process writes its local results to a separate file
- Good bandwidth, easy to implement
- Having many files makes data post process cumbersome
- **Can overwhelm the filesystem!**

  - Recall discussion of Lustre metadata servers
</div>

<div class="column">
![](img/posix-everybody.png)
</div>

# Programs with dedicated I/O processes ("I/O servers")

Variation of "spokesman": One or more MPI processes that *only* do I/O.

- Can be great for I/O-heavy programs. Used eg. by the aforementioned `ICON` code
- Pseudocode example:

<div class="column">
**Compute processes**
```
while (simulation_running)
    evolve_system()
    if (checkpoint)
        // Wait until our previous send
        // has been processed by the server
        MPI_WAIT(send_req)
        send_buffer = data
        MPI_ISend(IO_SERVER, send_buffer)
```
</div>

<div class="column">
**I/O server**
```
while (simulation_running)
    for (rank in compute_processes)
        MPI_Recv(rank, recv_buffer)
        write(buffer)
```
</div>


# Comments on standard I/O streams

The "standard" I/O streams `stdout`, `stdin`, `stderr` are effectively **serial** in `mpirun`/`srun` context!

- Ex: Default `srun` will redirect `stdout` of all processes to `stdout` of `srun`
- Do not rely on standard streams for heavy I/O operations. Code for direct file I/O instead (`fprintf` instead of `printf`)
- Avoid excessive debug prints in production runs
    - Separate "Debug" and "Release" builds if needed
