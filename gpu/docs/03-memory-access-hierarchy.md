---
title:    Memory Hierarchy and Memory Accesses in GPUs 
subtitle: High-Level GPU Programming 
date:     November 2024
lang:     en
---

# Memory hierarchy and memory accesses in GPUs{.section}

# Host memory and device memory

![](img/gpu-bws.png){.center width=65%}

- In most GPUs, the global GPU memory is physically distinct from the host (CPU) memory
- Data used by GPU kernels has to be copied from host to device (and back if data is needed also in CPU)
- Host-device bus (typically PCIe) has often low bandwidth
    - Can become performance bottleneck
- Memory copies can be done asynchronously with computations

# Unified/managed memory 

- Unified/managed memory is a single memory address space accessible from any processor in a system 
- GPU runtime and hardware automatically migrate memory pages to the memory of the accessing processor
- Makes programming easier
- Implicit memory copies may create performance issues
    - Memory copies should be profiled when using unified/managed memory 

# Global memory

<div class="column" style=width:68%>
- Accessible by all threads in a grid
- Slow, latency of eg. 600-700 cycles
    - Still, high bandwidth compared to CPU memory (1600 GB/s for a single GCD of AMD MI250X)
- Can be controlled by host (via pointer operations)
- Lifetime of the program
</div>

<div class="column" style=width:30%>
![](img/Grid_threads.png){.center width=80%}
</div>

# Coalesced memory access

<div class="column" style=width:68%>
- GPUs can typically access global memory via 32-, 64-, or 128-byte transactions
- When threads in a warp/wavefront operate on aligned elements close to each other, 
  memory loads and stores can be *coalesced*
- Local shared memory can be used to improve the memory accesses
</div>

<div class="column" style=width:30%>
![](img/sub_group.png){.center width=65%}
</div>

# Coalesced vs non-coalesced memory access

Parallel GPU code of `y=y+a*x`:

- Instances of the same function, **kernel**, running for different index `id`

```cpp
GPU_K void axpy_(int n, double a, double *x, double *y, int id)
{
        y[id] += a * x[id]; // Coalesced 
        y[id] += a * x[id*stride]; // Strided Non-Coalesced 
        y[id] += a * x[id+shift]; // Shifted (non-aligned) Non-Coalesced 
}

```
# Local shared memory

<div class="column">
- Accessible by all threads in a block
- Very fast memory, latency of eg. 6 cycles
- User programmable cache
- Lifetime of the thread block
</div>
<div class="column">
![](img/work_group.png){.center width=25%}
</div>

# Registers

- Fastest form of memory
- Private to each thread
- Local variables and intermediate results
- No cost at reading, write & read is costlier (*e.g.* 24-cycle latency)
- Lifetime of the kernel
- Not directly controllable by user
- When all registers are used, there is spilling of the local variables into the global memory.

# Summary

- In most GPUs, the GPU memory is physically distinct from the host (CPU) memory
    - frequent memory copies can become bottleneck
- Hierarchy of memories in GPU:
    - **global**: slowest, accessible by all threads
    - **local shared memory**: very fast, shared by threads within a block
    - **registers**: fastest, local to threads
- **coalesced access**: threads within a warp/wavefront access adjacent memory locations
- Local shared memory can be use a programmable cache

