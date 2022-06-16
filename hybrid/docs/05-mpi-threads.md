---
title:  Using MPI with multithreading
event:  CSC Summer School in High-Performance Computing 2022
lang:   en
---

# Thread support in MPI

![](img/mpi-thread-support.svg){.center width=80%}

# Thread safe initialization

`MPI_Init_thread(required, provided)`
  : `argc`{.input}, `argv`{.input}
    : Command line arguments in C
  : `required`{.input}
    : Required thread safety level
  : `provided`{.output}
    : Supported thread safety level
  : `error`{.output}
    : Error value; in C/C++ it's the return value of the function,
      and in Fortran an additional output parameter

- Pre-defined integer constants:
  <small>
  ```
  MPI_THREAD_SINGLE < MPI_THREAD_FUNNELED < MPI_THREAD_SERIALIZED < MPI_THREAD_MULTIPLE
  ```
  </small>


# Hybrid programming styles: fine/coarse grained

- Fine-grained
    - Use **omp parallel do/for** on the most intensive loops
    - Possible to hybridize an existing MPI code with little effort and in
      parts

- Coarse-grained
    - Use OpenMP threads to replace MPI tasks
    - Whole (or most of) program within the same parallel region
    - More likely to scale over the whole node, enables all cores to
      communicate (if supported by MPI implementation)


# Multiple thread communication

- Hybrid programming is relatively straightforward in cases where
  communication is done by only a single thread at a time
- With the "multiple" mode all threads can make MPI calls
  independently
  ```c
  int required=MPI_THREAD_MULTIPLE, provided;
  MPI_Init_thread(&argc, &argv, required, &provided)
  ```

- When multiple threads communicate, the sending and receiving threads
  normally need to match
    - Thread-specific tags
    - Thread-specific communicators

# Thread-specific tags

- In point-to-point communication the thread ID can be used to
  generate a tag that guides the messages to the correct thread

![](img/multiple-thread-communication.svg){.center width=60%}


# Thread-specific tags

- In point-to-point communication the thread ID can be used to
  generate a tag that guides the messages to the correct thread

```fortran
!$omp parallel private(tid, tidtag, ierr)
tid = omp_get_thread_num()
tidtag = 2**10 + tid

! mpi communication to the corresponding thread on another process
call mpi_sendrecv(senddata, n, mpi_real, pid, tidtag, &
                  recvdata, n, mpi_real, pid, tidtag, &
                  mpi_comm_world, stat, ierr)

!$omp end parallel
```


# Collective operations in the multiple mode

- MPI standard allows multiple threads to call collectives simultaneously
    - Programmer must ensure that the same communicator is not being
      concurrently used by two different collective communication calls at
      the same process
- In most cases, even with `MPI_THREAD_MULTIPLE` it is beneficial to
  perform the collective communication from a single thread (usually the
  master thread)
- Note that MPI collective communication calls do not guarantee
  synchronization of the thread order


# MPI thread support levels

- Modern MPI libraries support all threading levels
    - OpenMPI: Build time configuration, check with
    ```bash
    ompi_info | grep 'Thread support'
    ```
    - Intel MPI: When compiling with `-qopenmp` a thread safe version of the
      MPI library is automatically used
    - Cray MPI: Set `MPICH_MAX_THREAD_SAFETY` environment variable to
      `single`, `funneled`, `serialized`, or `multiple` to select the
      threading level
- Note that using `MPI_THREAD_MULTIPLE` requires the MPI library to
  internally lock some data structures to avoid race conditions
    - May result in additional overhead in MPI calls

# Summary

- Multiple threads may make MPI calls simultaneously
- Thread specific tags and/or communicators
- For collectives it is often better to use a single thread for
  communication

# Thread and process affinity{.section}

# Thread and process affinity

- Normally, operating system can run threads and processes in any
  logical core
- Operating system may even move the running process/thread from one core to another
    - Can be beneficial for load balancing
    - For HPC workloads often detrimental as private caches get
      invalidated and NUMA locality is lost
- User can control where tasks are run via affinity masks
    - Task can be *pinned* to a specific logical core or set of logical cores

# Controlling affinity

- Affinity for a *process* can be set with a `numactl` command
    - Limit the process to logical cores 0,3,7:
      <br>
      `numactl --physcpubind=0,3,7 ./my_exe`
    - Threads "inherit" the affinity of their parent process
- Affinity of a thread can be set with OpenMP environment variables
    - `OMP_PLACES=[threads,cores,sockets]`
    - `OMP_PROC_BIND=[true, close, spread, master]`
- OpenMP runtime prints the affinity with `OMP_DISPLAY_AFFINITY=true`

# Controlling affinity

```
export OMP_AFFINITY_FORMAT="Thread %0.3n affinity %A"
export OMP_DISPLAY_AFFINITY=true
./test
Thread 000 affinity 0-7
Thread 001 affinity 0-7
Thread 002 affinity 0-7
Thread 003 affinity 0-7
```


```
OMP_PLACES=cores ./test
Thread 000 affinity 0,4
Thread 001 affinity 1,5
Thread 002 affinity 2,6
Thread 003 affinity 3,7
```

# MPI+OpenMP thread affinity

<div class="column">
- MPI library must be aware of the underlying OpenMP for correct
  allocation of resources
    - Oversubscription of CPU cores may cause significant performance
      penalty
- Additional complexity from batch job schedulers
- Heavily dependent on the platform used!
</div>

<div class="column">
![](img/affinity.svg){.center width=70%}
</div>

# Slurm configuration at CSC

- Within a node, `--tasks-per-node` MPI tasks are spread
  `--cpus-per-task` apart
- Threads within a MPI tasks have the affinity mask for the
  corresponging
  <br>
  `--cpus-per-task` cores
```
export OMP_AFFINITY_FORMAT="Process %P thread %0.3n affinity %A"
export OMP_DISPLAY_AFFINITY=true
srun ... --tasks-per-node=2 --cpus-per-task=4 ./test
Process 250545 thread 000 affinity 0-3
...
Process 250546 thread 000 affinity 4-7
...
```

- Slurm configurations in other HPC centers can be very different
    - Always experiment before production calculations!

# Summary

- Performance of HPC applications is often improved when processes and
threads are pinned to CPU cores
- MPI and batch system configurations may affect the affinity
    - very system dependent, try to always investigate
