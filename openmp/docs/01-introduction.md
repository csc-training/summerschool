---
title:  Introduction to hybrid programming
event:  CSC Summer School in High-Performance Computing 2025
lang:   en
---

# Anatomy of supercomputer

- Supercomputers consist of nodes connected with high-speed network
    - Latency `~`1 µs, bandwidth `~`200 Gb / s
- A node can contain several multicore CPUS
- Additionally, nodes can contain one or more accelerators
- Memory within the node is directly usable by all CPU cores

<br>
 ![](img/supercomputer-anatomy.png){.center width=60%}

# Parallel programming models

- Parallel execution is based on threads or processes (or both) which
  run at the same time on different CPU cores
- Processes
    - Interaction is based on exchanging messages between processes
    - MPI (Message passing interface)
- Threads
    - Interaction is based on shared memory, i.e. each thread can
      access directly other threads data
    - OpenMP


# Parallel programming models

<!-- Presentation suggestion: discuss the analog of office worker using the same white board or communication with phones -->

 ![](img/processes-threads.png){.center width=80%}
<br>
<div class=column>
**MPI: Processes**

- Independent execution units
- Have their **own** memory space
- MPI launches N processes at application startup
- Works over multiple nodes
</div>
<div class=column>

**OpenMP: Threads**

- Threads **share** memory space
- Threads are created and destroyed (parallel regions)
- Limited to a single node
</div>

# Hybrid programming: Launch threads (OpenMP) *within* processes (MPI)

<div class="column">
  - Shared memory programming inside a node, message passing between
    nodes
  - Matches well modern supercomputer hardware
  - Optimum MPI task per node ratio depends on the application and should always be experimented.
</div>

<div class="column">
![](img/supercomputer-node-hybrid.png){.center width=80%}
</div>

# Potential advantages of the hybrid approach

- Fewer MPI processes for a given amount of cores
    - Improved load balance
    - All-to-all communication bottlenecks alleviated
    - Decreased memory consumption if implementation uses replicated
      data
- Additional parallelization levels may be available
- Possibility for dedicating threads to different tasks
    - e.g. a thread dedicated to communication or parallel I/O
- Dynamic parallelization patterns often easier to implement with OpenMP


# Disadvantages of hybridization

- Increased overhead from thread creation/destruction
- More complicated programming
    - Code readability and maintainability issues
- Thread support in MPI and other libraries needs to be considered

# Alternatives to OpenMP within a node

- pthreads (POSIX threads)
- Multithreading support in C++ 11
- Performance portability frameworks (SYCL, Kokkos, Raja)
- Intel Threading Building Blocks
