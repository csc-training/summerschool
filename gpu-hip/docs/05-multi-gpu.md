---
title:  Multi-GPU programming and HIP/OpenMP + MPI
event:  CSC Summer School in High-Performance Computing 2022
lang:   en
---

# Outline

* GPU context
* Device management
* Programming models
* GPU-GPU Peer access (HIP)
* HIP/OpenMP + MPI


# Introduction

* Workstations or supercomputer nodes can be equipped with several GPUs
    * For the current supercomputers, the number of GPUs per node usually
      ranges between 2 to 6
    * Allows sharing (and saving) resources (disks, power units, e.g.)
    * More GPU resources per node, better per-node-performance


# GPU Context

* Context is established implicitly on the current device when the first task requiring an active context is evaluated (HIP and OpenMP)
* Several processes can create contexts for a single device
* The device resources are allocated per context
* By default, one context per device per process in HIP (since CUDA 4.0)
    * Threads of the same process share the primary context (for each device)
* HIP supports explicit context management 
* OpenMP does not support explicit context management


# Selecting device

* Driver associates a number for each available GPU device starting from 0
* The functions `hipSetDevice()` and `omp_set_default_device()` are used for selecting the desired device for HIP and OpenMP, respectively


# Device management

```cpp
// Return the number of available devices

int omp_get_num_devices(void); // OpenMP, returns the result
hipError_t hipGetDeviceCount(int *count) // HIP, stores the result in `count`

// Set device as the current device for the calling host thread

void omp_set_default_device(int device) // OpenMP
hipError_t hipSetDevice(int device) // HIP

// Return the current device for the calling host thread

int omp_get_default_device(void) // OpenMP, returns the result
hipError_t hipGetDevice(int *device) // HIP, stores the result in `device`
```


# Querying or verifying device properties

* In HIP, one can query the properties of different devices in the system using
  `hipGetDeviceProperties()` function
    * No context needed
    * Provides e.g. name, amount of memory, warp size, support for unified
      virtual addressing, etc.
    * Useful for code portability

In HIP, the function returns the device properties in struct `prop`
```
hipError_t hipGetDeviceProperties(struct hipDeviceProp *prop, int device)
```
In OpenMP, `requires` clause can be used to verify the device properties, eg,
```
#pragma omp requires unified_shared_memory
```



# Multi-GPU programming models

<div class="column">
* One GPU per process
    * Syncing is handled through message passing (eg. MPI)
* Many GPUs per process
    * Process manages all context switching and syncing explicitly
* One GPU per thread
    * Syncing is handled through thread synchronization requirements
</div>

<div class="column">
![](img/single_proc_mpi_gpu2.png){width=50%}
![](img/single_proc_multi_gpu.png){width=50%}
![](img/single_proc_thread_gpu.png){width=50%}
</div>


# Multi-GPU, one GPU per process

* Recommended for multi-process applications using a message passing library
* Message passing library takes care of all GPU-GPU communication
* Each process interacts with only one GPU which makes the implementation
  easier and less invasive (if MPI is used anyway)
    * Apart from each process selecting a different device, the implementation
      looks much like a single-GPU program
* **Multi-GPU implementation using MPI is discussed at the end!**


# Multi-GPU, many GPUs per process

* Process switches the active GPU using `hipSetDevice()` (HIP) or `omp_set_default_device()` (OpenMP) functions
* After setting the device, operations such as the following are effective only
  on the selected GPU:
    * Memory operations
    * Kernel execution
    * Streams and events (HIP)
* Asynchronous function calls calls (HIP) or `nowait` clause (OpenMP) are required to overlap work 

# Multi-GPU, one GPU per thread

* One GPU per CPU thread
    * I.e one OpenMP CPU thread per GPU being used
* HIP API is threadsafe
    * Multiple threads can call the functions at the same time
* Each thread can create its own context on a different GPU
    * `hipSetDevice()` or `omp_set_default_device()` set the device and create a context per thread
    * Easy device management with no changing of device
* Communication between threads becomes a bit more tricky

# Peer access (HIP)

* Access peer GPU memory directly from another GPU
    * Pass a pointer to data on GPU 1 to a kernel running on GPU 0
    * Transfer data between GPUs without going through host memory
    * Lower latency, higher bandwidth

```cpp
// Check peer accessibility
hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice)

// Enable peer access
hipError_t hipDeviceEnablePeerAccess(int peerDevice, unsigned int flags)

// Disable peer access
hipError_t hipDeviceDisablePeerAccess(int peerDevice)
```
* Between AMD GPUs, the peer access is always enabled


# Peer to peer communication

* Devices have separate memories
* Memcopies between different devices can be done as follows:

```cpp
// HIP: First option that requires unified virtual addressing
hipError_t hipMemcpy(void* dst, void* src, size_t size, 
hipMemcpyKind kind=hipMemcpyDefault)

// HIP: Second option does not require unified virtual addressing
hipError_t hipMemcpyPeer(void* dst, int  dstDev, void* src, int srcDev, size_t size)

// OpenMP
int omp_target_memcpy(void *dst, const void *src, size_t size, size_t dstOffset,
                      size_t srcOffset, int dstDev, int dstDev);
```

* If peer to peer access is not available, the functions result in a normal
  copy through host memory

# Message Passing Interface (MPI)

* MPI is a widely adopted standardized message passing interface for
  distributed memory parallel computing
* The parallel program is launched as a set of independent, identical
  processes
    * Same program code and instructions
    * Each process can reside in different nodes or even different computers
* All variables and data structures are local to the process
* Processes can exchange data by sending and receiving messages


# Three levels of parallelism

1. GPU - GPU threads on the multiprocessors
    * Parallelization strategy: HIP, SYCL, Kokkos, OpenMP
2. Node - Multiple GPUs and CPUs
    * Parallelization strategy: MPI, Threads, OpenMP
3. Supercomputer - Many nodes connected with interconnect
    * Parallelization strategy: MPI between nodes

![](img/parallel_regions.png){width=40%}


# MPI and HIP

* Compiling HIP/OpenMP and MPI calls in the same compilation unit may not always be trivial
* One can set MPI compiler to use `hipcc` or the desirable OpenMP compiler like `nvc`, eg for OpenMPI:
```bash
OMPI_CXXFLAGS='' OMPI_CXX='hipcc'
```
or 
```bash
OMPI_CXXFLAGS='' OMPI_CXX='nvc -mp=gpu -gpu=cc80'
```

* Alternatively, one could separate HIP/OpenMP and MPI code in different compilation units compiled with `mpicxx` and `hipcc` or `nvc`
    * Link object files in a separate step using `mpicxx` or `hipcc`/`nvc`


# MPI+HIP strategies

1. One MPI process per node
2. **One MPI process per GPU**
3. Many MPI processes per GPU, only one uses it
4. **Many MPI processes sharing a GPU**

* 2 is recommended (also allows using 4 with services such as CUDA MPS)
    * Typically results in most productive and least invasive implementation
      for an MPI program
    * No need to implement GPU-GPU transfers explicitly (MPI handles all
      this)
    * It is further possible to utilize remaining CPU cores with OpenMP (but
      this is not always worth the effort/increased complexity)


# Selecting the correct GPU

* Typically all processes on the node can access all GPUs of that node
* The following implementation allows utilizing all GPUs using one or more
  processes per GPU
    * Use CUDA MPS when launching more processes than GPUs

```cpp
int deviceCount, nodeRank;
MPI_Comm commNode;
MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &commNode);
MPI_Comm_rank(commNode, &nodeRank);
#ifdef _OPENMP
  deviceCount = omp_get_num_device();
  omp_set_default_device(nodeRank % deviceCount)
#elif __HIP__
  hipGetDeviceCount(&deviceCount);
  hipSetDevice(nodeRank % deviceCount);
#endif
```


# GPU-GPU communication through MPI

* CUDA/ROCm aware MPI libraries support direct GPU-GPU transfers
    * Can take a pointer to device buffer (avoids host/device data copies)
* Unfortunately, currently no GPU support for custom MPI datatypes (must use a
  datatype representing a contiguous block of memory)
    * Data packing/unpacking must be implemented application-side on GPU
* ROCm aware MPI libraries are under development and there may be problems
    * It is a good idea to have a fallback option to use pinned host staging
      buffers


# Summary

- There are many options to write a multi-GPU program
- Use `hipSetDevice()` or `omp_set_default_device()` to select the device, and the subsequent tasks
  operate on that device
* If you have an MPI program, it is often best to use one GPU per process, and
  let MPI handle data transfers between GPUs
* There is still little experience from ROCm aware MPIs, there may be issues
    * Note that a CUDA/ROCm aware MPI is only required when passing device
      pointers to the MPI, passing only host pointers does not require any
      CUDA/ROCm awareness
