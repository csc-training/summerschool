---
title:    Memory Hierarchy and Memory Accesses in GPUs
author:   CSC Training
date:     2025-06
lang:     en
---

# Memory hierarchy{.section}

# Memory hierarchy

| type          | size          | latency           | accessibility             |
|---------------|---------------|-------------------|---------------------------|
| host          | 10-100 GB     | >10k cycles       | host (sometimes device)   |
| device global | 10-100 GB     | 100-1000 cycles   | entire device (grid)      |
| device local  | 10-100 kB     | 1-10 cycles       | CU (block)                |
| registers     | 10-100 kB     | 1 cycle           | SIMD (warp/wavefront)     |

# Global memory

- Accessible by all threads in a grid
- Slow, latency of eg. 600-700 cycles
    - Still, high bandwidth compared to CPU memory (1600 GB/s for a single GCD of AMD MI250X)
- Can be controlled by host (via pointer operations)
- Lifetime of the program

# Local shared memory

- Accessible by all threads in a block (local to the CU)
- Very fast memory, latency of eg. 6 cycles
- User programmable cache
    - Load frequently used values once from global memory, store in local memory
    - Useful when touching the same memory multiple times inside a block
- Lifetime of the thread block

# Registers

- Fastest form of memory
- Private to each thread
- Local variables and intermediate results
- No cost at reading, write & read is costlier (*e.g.* 24-cycle latency)
- Lifetime of the kernel
- Not directly controllable by user
- When all registers are used, there is spilling of the local variables into the global memory.

# Memory access patterns{.section}

# Coalesced memory access

- GPUs can typically access global memory via 32-, 64-, or 128-byte transactions
- When threads in a warp/wavefront operate on aligned elements close to each other, 
  memory loads and stores can be *coalesced*
    - Coalesced: multiple threads are served by a single memory operation
    - Non-coalesced: multiple memory accesses needed to serve multiple threads

# Coalesced vs non-coalesced memory access

```cpp
// Linear index across the entire grid
const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

coalesced[tid] = 2.0f;          // 2 x 128 byte transactions for 64 threads
non_coalesced[4 * tid] = 2.0f;  // 8 x 128 byte transactions for 64 threads
```
```
Coalesced
    tid      0   1   2   3 
    maps to  |   |   |   | 
    address [0] [1] [2] [3]

Non-coalesced
    tid      0   1   2   3_____________________________________
    maps     |   |   |_______________________                  |
    to       |   |___________                |                 |
             |               |               |                 |
    address [0] [1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12] [13] [14] [15]
```

# Summary

- In most GPUs, the GPU memory is physically distinct from the host (CPU) memory
    - frequent memory copies can become bottleneck
- Hierarchy of memories in GPU:
    - **global**: slowest, accessible by all threads
    - **local shared memory**: very fast, shared by threads within a block
    - **registers**: fastest, local to threads
- **coalesced access**: threads within a warp/wavefront access adjacent memory locations
- Local shared memory can be use a programmable cache

