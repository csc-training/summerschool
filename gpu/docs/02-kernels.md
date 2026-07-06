---
# SPDX-FileCopyrightText: 2026 CSC - IT Center for Science Ltd. <www.csc.fi>
#
# SPDX-License-Identifier: CC-BY-4.0

title:  HIP, CUDA and GPU kernels
event:  CSC Summer School in High-Performance Computing 2026
lang:   en
---

# Overview

We will cover the following topics
  \
  \

- CUDA/HIP terminology
- Some APIs and keywords
- Thread hierarchy
- (Basic) Memory management
- Error checking


# Learning Objectives

After this lecture you will understand
  \
  \
  
- GPU terminology (e.g. what is a thread, a kernel,...)
- Clarifying GPU programming model 

But most important, you will be able to launch kernels!


# CUDA

::: incremental
- Software stack offering tools for programming Nvidia GPUs
- CUDA includes C++ runtime API **and** a kernel programming language
- Standard C++ syntax, `nvcc` compiler driver is used to compile code
    - `nvcc` uses a CPU compiler, e.g. `g++` in the background for compiling CPU code
- Includes libraries for numerical maths
:::

# ROCm

::: incremental
- Software stack offering tools for programming AMD GPUs
- ROCm provides the tools for HIP, OpenCL and OpenMP
    - Compilers, libraries for high-level functions, debuggers, profilers and runtimes
- ROCm is **not** a programming language
- Includes libraries for numerical maths
:::


# HIP

::: incremental
- HIP = Heterogeneous-computing Interface for Portability
- AMD effort to offer a common programming interface that works on both
      Nvidia and AMD devices
- Almost a one-to-one clone of CUDA from the user perspective
- Standard C++ syntax, `hipcc` wrapper for compiling
    - Uses `nvcc`/`clang++` compilers behind the scenes
- Allows one to write portable GPU code
:::

# GPU terminology

::: incremental
- Compute unit (CU, AMD) / streaming multiprocessor (SM, Nvidia)
    - A simple processor on a GPU
    - Contains multiple independent vector units
- Kernel
    - Parallel function executed on the GPU
- Thread
    - Individual worker of a wavefront/warp (AMD/Nvidia)
:::

# GPU terminology

::: incremental
- Wavefront/Warp (AMD/Nvidia)
    - Collection of threads: execute the same instruction in lockstep
    - Fixed number of threads (AMD: 64, NVIDIA 32)
    - Threads per block is chosen at kernel launch
        - Wavefronts per block = threads per block / 64
- workgroup / Thread block (AMD/Nvidia)
    - Group of threads partitioned to wavefronts/warps
    - Execute on the same CU/SM (AMD/Nvidia)
    - Can synchronise together and communicate through memory in the CU/SM (AMD/Nvidia)
:::

# HIP/CUDA programming model

::: incremental
- GPU accelerator is often called a *device* and CPU a *host*
- Parallel GPU code is...
    - ...launched by the host using the API, ...
    - ...is written using the kernel language...
    - ...from the point of view of a single thread, and...
    - ...is executed on a device by many threads.
:::

# GPU programming considerations

::: incremental
- The parallel nature of GPUs requires many similar tasks that can be executed simultaneously
    - One usage is to replace iterations of loop with a GPU kernel call
- Need to adapt CPU code to run on the GPU
    - Algorithmic changes to fit the parallel execution model
    - Share data among hundreds of cooperating threads
    - Manage data transfers between CPU and GPU memories
      carefully (a common bottleneck)
:::

# API

Code on the CPU to control the larger context and the flow of execution

::: incremental
- Device init and management: `hipSetDevice`/`cudaSetDevice`
- Memory management: `hipMalloc`/`cudaMalloc`
- Execution control: `kernel<<<blocks, threads>>>`
- Synchronisation: device, stream, events: `hipDeviceSynchronize`/`cudaDeviceSynchronize`
- Error handling, context handling, ... : `hipGetErrorString`/`cudaGetErrorString`

- Documentation: [HIP docs](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules.html#modules-reference) & [CUDA docs](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
:::

# API example: Hello world

```cpp
#include <hip/hip_runtime.h>
#include <stdio.h>

int main(void)
{
    int count = 0;
    auto result = hipGetDeviceCount(&count);

    int device = 0;
    result = hipGetDevice(&device);

    printf("Hello! I'm GPU %d out of %d GPUs in total.\n", device, count);

    return 0;
}
```

# Kernels

Code on the GPU from the point of view of a single thread

::: incremental
- kernel is a function executed by the GPU
- kernel must be declared with the `__global__` attribute and the return type must be `void`
- any function called from a kernel must be declared with `__device__` attribute
- all pointers passed to a kernel should point to memory accessible from
  the device
- unique thread and block IDs can be used to distribute work
:::

# Kernel language

::: incremental
- attributes: `__device__`, `__global__`, `__shared__`, ...
- built-in variables: `threadIdx.x`, `blockIdx.y`, ...
- vector types: `int3`, `float2`, `dim3`, ...
- math functions: `sqrt`, `powf`, `sinh`, ...
- atomic functions: `atomicAdd`, `atomicMin`, ...
- intrinsic functions: `__syncthreads`, `__threadfence`, ...
:::

# Kernel example: $\vec{y} = \alpha \vec x + \vec y$ (axpy)

```cpp
__global__ void axpy(int n, double a, double *x, double *y) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        y[tid] += a * x[tid];
    }
}
```
![](img/kernel_tid_limit.svg){.center width=70%}

- global ID `tid` calculated based on the thread and block IDs
- only threads with `tid` smaller than `n` calculate
- works only if number of threads ≥ `n`


# Kernel example: axpy (revisited)

```cpp
__global__ void axpy(int n, double a, double *x, double *y) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        y[i] += a * x[i];
    }
}
```
![](img/kernel_stride_limit.svg){.center width=70%}

- handles any vector size, but grid size should still be chosen with some care

# Launching kernels

::: incremental
- kernels are launched with the following syntax:
  ```cpp
  axpy<<<blocks, threads, shmem, stream>>>(args)
  ```

- grid dimensions are obligatory
    - must have an integer type or vector type of `dim3`
- `shmem`, and `stream` are optional arguments 
- kernel execution is asynchronous with the host
:::

# Grid: thread hierarchy

Kernels are executed over a grid:

- grid consists of equisized blocks of threads
- grid size & dimensionality is set at kernel launch
```cpp
// num blocks in grid
dim3 blocks(10, 1, 1);

// num threads in block
dim3 threads(1024, 1, 1);

// 10 blocks in a grid, every block is the same size (1024).
// In total 10 x 1024 = 10240 threads over the entire grid.

// Launch 'axpy' over a grid defined by 'blocks' and 'threads'
// with arguments 'args'
axpy<<<blocks, threads>>>(args)
```

# Grid: thread hierarchy

- threads execute the kernel, grid describes the size & dimensions
- built-in variables in a kernel: `threadIdx`, `blockIdx`, `blockDim`, `gridDim` relate to the dim3 variables used at kernel launch
```cpp
__global__ void axpy(int n, double a, double *x, double *y)
{
    // If launched with:
    // dim3 blocks(10, 1, 1);
    // dim3 threads(1024, 1, 1);
    // axpy<<<blocks, threads>>>(...);

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //              ^ [0, 1023]  ^ [0, 9]      ^ 1024 
    const int stride = blockDim.x * gridDim.x;
    //                 ^ 1024       ^ 10
}
```

# Memory management

- GPU has its own memory area
- allocate device usable memory with `hipMalloc`/`cudaMalloc` (cf. `std::malloc`)
- pass the pointer to the kernel
\ 
\ 
```cpp
const size_t num_bytes = sizeof(double) * n;
void *dx = nullptr;
hipMalloc(&dx, num_bytes);
```

# Memory management

- copy data to/from device: `hipMemcpy`/`cudaMemcpy` (cf. `std::memcpy`)
\ 
\ 
```cpp
// Recommended
// Implicit copy direction, runtime figures it out
// from the virtual address of the pointer.
hipMemcpy(dx, x, num_bytes, hipMemcpyDefault);
hipMemcpy(x, dx, num_bytes, hipMemcpyDefault);

// Explicit copy direction with the 'kind' parameter
// Not recommended: giving the wrong direction
// is just silently ignored
hipMemcpy(dx, x, num_bytes, hipMemcpyHostToDevice);
hipMemcpy(x, dx, num_bytes, hipMemcpyDeviceToHost);
```

# Error checking

- always use error checking with larger codebases!
  - it has low overhead, and can save a lot of debugging time!
- some exercises of this course do not have error checking, mostly to focus on the taught topic

```cpp
#define HIP_ERRCHK(result) hip_errchk(result, __FILE__, __LINE__)
static inline void hip_errchk(hipError_t result, const char *file, int line) {
    if (result != hipSuccess) {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(result), file, line);
        exit(EXIT_FAILURE);
    }
}

// Wrap API call with the macro
HIP_ERRCHK(hipMalloc(&ptr, bytes));

```

# Error checking kernel launch

- kernel launches may fail due to bad conf param: catch with macro!

```cpp
#define LAUNCH_KERNEL(kernel, ...) launch_kernel(#kernel, __FILE__, __LINE__, kernel, __VA_ARGS__)
template <typename... Args>
void launch_kernel(const char *kernel_name, const char *file, int32_t line,
                   void (*kernel)(Args...), dim3 blocks, dim3 threads,
                   size_t num_bytes_shared_mem, hipStream_t stream, Args... args) {
    auto result = hipGetLastError();
    kernel<<<blocks, threads, num_bytes_shared_mem, stream>>>(args...);
    result = hipGetLastError();
    if (result != hipSuccess) {
        printf("Error with kernel \"%s\" in %s at line %d\n%s: %s\n",
               kernel_name, file, line, hipGetErrorName(result),
               hipGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}
```

# Summary

::: incremental
- HIP supports both AMD and NVIDIA devices
- CUDA and HIP consist of an API and a kernel language
    - API controls the larger context
    - kernel language for single thread point of view GPU code
- kernels execute over a grid of (blocks of) threads
    - each block is executed in wavefronts of 64 (AMD) or 32 (NVIDIA) threads
- kernels need to be declared `__global__ void` and are launched with `kernel<<<blocks, threads>>>(arguments)`
:::
