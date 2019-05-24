---
title:  "OpenACC: advanced topics"
author: CSC Summerschool
date:   2019-07
lang:   en
---


# Multi-GPU programming with OpenACC

- Hardware
    - Typical GPU node in a cluster
- Programming models
    - Single thread
    - Threads with OpenMP
    - Multiple processes with MPI


# Typical HPC GPU cluster

- Three levels of hardware parallelism
    - GPU - different levels of threads
    - Node - GPU, CPU and interconnect
    - Machine - several nodes connected with interconnect

FIXME: missing figure


# Typical HPC GPU cluster

- Parallelization
    - OpenACC
    - Threads (pthreads, OpenMP, TBB) or MPI for CPUs
    - MPI between nodes

FIXME: missing figure


# Multi-GPU communication cases

- All GPUs of a node are accessible from a single process and thread
    - Data copies either directly or through CPU memory
- Communication between nodes require message passing

FIXME: missing table


# Single node {.section}


# Multi-GPU programming with OpenACC

- OpenACC permits using multiple GPUs within one node by using the
  `acc_get_num_devices` and `acc_set_device_num` functions
- Asynchronous OpenACC calls, OpenMP threads or MPI processes must be used
  in order to actually run kernels in parallel
- Data transfers between GPUs must be done via CPU memory manually
    - You can also use CUDA features or managed memory


# Example

```c
for (int block = 0; block < blocks; block++) {
    int first_row = block * blocksize;
    int last_row = first_row + blocksize;
    acc_set_device_num(block%gpus, acc_device_nvidia);
#pragma acc parallel loop private(ii, ir, Ci, Cr) async(block%gpus)
    for (ii = first_row; ii < last_row; ii++) {
        ...
    }
#pragma acc update self(values[first_row:blocksize][0:cols]) async(block%gpus)
}
for (int gpu = 0; gpu < gpus; gpu++) {
    acc_set_device_num(gpu, acc_device_nvidia);
#pragma acc wait
}
```


# OpenMP and OpenACC

```c
#pragma omp parallel private(first_row, last_row, gpu)
{
    gpu = omp_get_thread_num();  // Assign one thread per GPU
    acc_set_device_num(gpu, acc_device_nvidia);
    first_row = gpu  blocksize;
    last_row = first_row + blocksize;
    #pragma acc data copyout(iterations[first_row:blocksize][0:cols])
    {
        #pragma acc parallel loop private(ii, ir, Ci, Cr)
        for (ii = first_row; ii < last_row; ii++) {
            ...
        }
    }
}
```


# MPI and OpenACC {.section}


# Message-passing interface

- MPI is an application programming interface (API) for communication
  between separate processes
    - The most widely used approach for *distributed* parallel computing
- Parallel program is launched as set of independent, identical processes
- The same program code and instructions
- Can reside in different nodes
    - or even in different computers


# MPI ranks

- MPI runtime assigns each process a rank
    - identification of the processes
    - ranks start from 0 and extent to N-1
- Processes can perform different tasks and handle different data basing
  on their rank

FIXME: missing code


# Data model

- All variables and data structures are local to the process
- Processes can exchange data by sending and receiving messages

FIXME: missing figure


# MPI communicator

- Communicator is an object connecting a group of processes
- Initially, there is always a communicator MPI_COMM_WORLD which
  contains all the processes
- Most MPI functions require communicator as an argument
- Users can define own communicators


# MPI Hello world!

```c
#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>

int main(int argc, char argv[])
{
    int i, myid, ntasks;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if (myid == 0) {
        printf("In total there are %i tasks\n", ntasks);
    }
    printf("Hello from %i\n", myid);
    MPI_Finalize();
    return 0;
}
```


# Multi-GPU programming with MPI+OpenACC

- Idea: use MPI to transfer data between GPUs, use OpenACC-kernels for
  computations
- Additional complexity: GPU memory is separate from that of a CPU
    - Without GPU-aware MPI-library: data must be transferred from the
      device memory to the host memory and vice versa before performing
      MPI-calls
    - With GPU-aware MPI-library: data on the device can be directly used
      in MPI-calls (via transparent RDMA)


# MPI communication without GPU-aware MPI

- MPI send
    1. Copy data from the device to a buffer on the host
    2. Send the data from the buffer on the **host** with MPI
- MPI receive
    1. Receive the data to a buffer on the **host** with MPI
    2. Copy data from the buffer on the host to the device
- To perform point-to-point communication, two additional buffers and data
  transfers needed


# MPI communication without GPU-aware MPI

```c
/* MPI_Send without GPU-aware MPI */
#pragma acc update host(data[0:N])
/* Send and receive */
MPI_Send(data, N, MPI_DOUBLE, to, MPI_ANY_TAG, MPI_COMM_WORLD);

/* MPI_Recv without GPU-aware MPI */
MPI_Recv(data, N, MPI_DOUBLE, from, MPI_ANY_TAG, MPI_COMM_WORLD,
         MPI_STATUS_IGNORE);
/* Copy data from host to GPU */
#pragma acc update device(data[0:N])
```


# GPU aware MPI

- Can use the device pointer in MPI calls - no need for additional buffers
    - No need for extra buffers and device-host-device copies

FIXME: missing figure


# Using device addresses with host_data

- For accessing device addresses of data on the host OpenACC includes
  **host_data** --construct with the **use_device** --clause
- No additional data transfers needed between the host and the device,
  data automatically accessed from the device memory via **R**emote
  **D**irect **M**emory **A**ccess
- Requires *library* and *device* support to function!


# MPI communication with GPU-aware MPI

- MPI send
    - Send the data from the buffer on the **device** with MPI
- MPI receive
    - Receive the data to a buffer on the **device** with MPI
- No additional buffers or data transfers needed to perform
  communication


# MPI communication with GPU-aware MPI

```c
/* MPI_Send with GPU-aware MPI */
#pragma acc host_data use_device(data)
{
    MPI_Send(data, N, MPI_DOUBLE, to, MPI_ANY_TAG, MPI_COMM_WORLD);
}

/* MPI_Recv with GPU-aware MPI */
#pragma acc host_data use_device(data)
{
    MPI_Recv(data, N, MPI_DOUBLE, from, MPI_ANY_TAG, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
}
```

# GPU affinity {.section}


# Selecting the GPU

- Problem description:
    - If a node has more than one GPU, all processes in the node can
      access all GPUs of the node
    - MPI processes do not have a priori information on the other ranks in
      the same node
    - Which GPU the MPI process should select?

FIXME: missing figure


# Selecting the GPU

- Simple approach
- Use batch system options to assign the ranks in linear fashion to the
  cores
    - Node 1 ranks: 0 1 2 3 4 5
    - Node 2 ranks: 6 7 8 9 10 11
- You may also assume that you know the number of processes per node to
  compute the rank on the node
    - **int nodeRank = rank % processesPerNode**
- Another non-portable option is to use the environment variables of the
  batch job system


# Selecting the GPU (cont.)

- Simple approach is not very robust
- More portable and robust solution is to use MPI3 shared memory
  communicators:

```c
MPI_Comm shared;
int local_rank, local_size, num_gpus;

MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                    MPI_INFO_NULL, &shared);
MPI_Comm_size(shared, &local_size); // number of ranks in this node
MPI_Comm_rank(shared, &local_rank); // my local rank
num_gpus = acc_get_num_device(acc_device_nvidia); // num of gpus in node
if (num_gpus == local_size) {
    acc_set_device_num(local_rank);
} // otherwise error
```

# Summary

- Typical HPC cluster node has several GPUs in each node
- Different programming models for shared and distributed memory
    - Threads within a node
    - Message passing between nodes
- Selecting the GPUs with correct affinity


# Routine directive {.section}


# Function calls in compute regions

- Often it can be useful to call functions within loops to improve
  readability and modularisation
- By default OpenACC does not create accelerated regions for loops
  calling functions
- One has to instruct the compiler to compile a device version of the
  function


# Routine directive

- Define a function to be compiled for an accelerator as well as the host
    - C/C++: `#pragma acc routine (name) [clauses]`
    - Fortran: `!$acc routine (name) [clauses]`
- The directive should be placed at the function declaration
    - Visible both to function definition (actual code) and call site
- Optional name enables the directive to be declared separately


# Routine directive

- Clauses defining level of parallelism in function
    - `gang` Function contains gang level parallelism
    - `worker` Function contains worker level parallelism
    - `vector` Function contains vector level parallelism
    - `seq` Function is not OpenACC parallel
- Other clauses
    - `nohost` Do not compile host version
    - `bind(string)` Define name to use when calling function in
      accelerated region


# Routine directive example

<div class="column">
## C/C++
```c
#pragma acc routine vector
void foo(float* v, int i, int n) {
    #pragma acc loop vector
    for ( int j=0; j<n; ++j) {
        v[i*n+j] = 1.0f/(i*j);
    }
}

#pragma acc parallel loop
for (int i=0; i<n; ++i) {
    foo(v,i);
    // call on the device
}
```

- Example from
https://devblogs.nvidia.com/parallelforall/7-powerful-new-features-openacc-2-0/
</div>

<div class="column">
## Fortran
```fortran
subroutine foo(v, i, n)
  !$acc routine vector
  real :: v(:,:)
  integer :: i, n
  !$acc loop vector
  do j=1,n
     v(i,j) = 1.0/(i*j)
  enddo
end subroutine

!$acc parallel loop
do i=1,n
  call foo(v,i,n)
enddo
!$acc end parallel loop
```
</div>


# Asynchronous operations {.section}


# Motivation

- By default, the local thread will wait until OpenACC compute or data
  construct has completed its execution
- Potential parallelism in overlapping compute, data transfers, MPI,
  etc.

FIXME: missing figure


# Asynchronous execution: async clause

- `async[(int-expr)]` clause enables one to enqueue compute and
  data operations, and local (host) thread will continue execution
- Order is preserved for enqueued operations
- OpenACC `async` clause is supported by constructs
    - `parallel`, `kernels`
    - `enter data, exit data, update`
    - `wait`

# Asynchronous execution: wait directive

- Causes the CPU thread to wait for completion of asynchronous
  operations
    - `#pragma acc wait [(int-expr-list)] [clauses]
    - !$acc wait [(int-expr-list)] [clauses]`


# OpenACC and asynchronous execution

FIXME: missing figure


# Multiple queues

- One can have multiple queues, enabling one to overlap execution of
  kernels and data operations
- `async` clause
    - non-negative integer argument, defining on which queue the operation
      is placed
    - Within one queue order is preserved, in different queues operations
      have no coupling
    - If no queue is given the default queue is used


# Multiple queues

- One can have multiple queues, enabling one to overlap execution of
  kernels and data operations
- `wait` directive
    - list of integers as argument, defining which queues to wait on.
    - By default it waits for all.


# OpenACC and asynchronous execution

FIXME: missing figure


# Example c = a + b

```c
a = malloc(sizeof(double) * N);
b = malloc(sizeof(double) * N);
c = malloc(sizeof(double) * N);

for (int i = 0; i < N;i++) {
    a[i] = i;
    b[i] = i;
}

#pragma acc data create(a[:N], b[:N], c[:N])
{
    t1 = omp_get_wtime();
    for(q = 0; q < queues; q++) {
        qLength = N / queues;
        qStart = q  qLength;
        #pragma acc update device(a[qStart:qLength], b[qStart:qLength]) async(q)
        #pragma acc parallel loop async(q)
        for (int i = qStart; i < qStart + qLength; i++) {
            c[i] = a[i] + b[i];
        }

        #pragma acc update self(c[qStart:qLength]) async(q)
    } //end for (q)
    #pragma acc wait
    t2 = omp_get_wtime();
} //end acc data

printf("compute in %g sn", t2 - t1);
```


# Interoperability with libraries {.section}


# Interoperability with libraries

- Often it may be useful to integrate the accelerated OpenACC code with
  other accelerated libraries
- MPI: MPI libraries are CUDA-aware
- CUDA: It is possible to mix OpenACC and CUDA
    - Use OpenACC for memory management
    - Introduce OpenACC in existing GPU code
    - Use CUDA for tightest kernels, otherwise OpenACC
- Numerical GPU libraries: CUBLAS, CUFFT, MAGMA, CULA...
- Thrust, etc.


# Device data interoperability

- OpenACC includes methods to access to device data pointers
- Device data pointers can be used to interoperate with libraries and
  other programming techniques available for accelerator devices
    - CUDA kernels and cuBLAS libraries
    - CUDA-aware MPI libraries
- Some features are still under active development, many things may not
  yet work as they are supposed to!


# Data constructs: `host_data`

- Define a device address to be available on the host
    - C/C++: `#pragma acc host_data [clause]`
    - Fortran:`!$acc host_data [clause]`
- Only a single clause is allowed: C/C++, Fortran: `use_device(var-list)`
- Within the construct, all the variables in `var-list` are referred
  to by using their device addresses


# host_data construct: example with cublas

```c
cublasInit();
double x,y;
//Allocate and initialise x
#pragma acc data copyin(x[:n]) copy(y[:n])
{
    #pragma acc host_data use_device(x,y) {
        cublasDaxpy(n, a, x, 1, y, 1);
    } // #pragma acc host_data
} // #pragma acc data
}
```

- Using PGI/OpenACC compiler, the cuBLAS can be accessed by providing the
  library to the linker : `-L$CUDA_INSTALL_PATH/lib64 --lcublas`
- Recent versions: `-Mcudalib=cublas`


# Calling CUDA-kernel from OpenACC-program

- In this scenario we have a (main) program written in C/C++ (or Fortran)
  and this driver uses OpenACC directives
    - CUDA-kernels must be called with help of OpenACC `host_data`
- Interface function in CUDA-file must have `extern "C" void func(...)`
- The CUDA-codes are compiled with NVIDIA `nvcc` compiler, e.g.
  `nvcc -c -O4 --restrict -arch=sm_35 daxpy_cuda.cu`
- The OpenACC-codes are compiled with PGI-compiler e.g.
  `pgcc -c -acc -O4 call_cuda_from_openacc.c`
- Linking with PGI-compiler must also have `-acc -Mcuda` e.g.
  `pgcc -acc -Mcuda call_cuda_from_openacc.o daxpy_cuda.o`


# Calling CUDA-kernel from OpenACC-program

<div class="column">
```c
// call_cuda_from_openacc.c

extern void daxpy(int n, double a,
                  const double *x, double *y);

int main(int argc, char *argv[])
{
    int n = (argc > 1) ? atoi(argv[1]) : (1 << 27);
    const double a = 2.0;
    double *x = malloc(n * sizeof(*x));
    double *y = malloc(n * sizeof(*y));

    #pragma acc data create(x[0:n], y[0:n])
    {
        // Initialize x & y
        ...
        // Call CUDA-kernel
        #pragma acc host_data use_device(x,y)
        daxpy(n, a, x, y);
        ...
    } // #pragma acc data
```
</div>

<div class="column">
```c
// daxpy_cuda.cu

__global__
void daxpy_kernel(int n, double a,
                  const double *x, double *y)
{ // The actual CUDA-kernel
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n) {
        y[tid] += a * x[tid];
        tid += stride;
    }
}
extern "C" void daxpy(int n, double a,
                      const double *x, double *y)
{ // This can be called from C/C++ or Fortran
    dim3 blockdim = dim3(256,1,1);
    dim3 griddim = dim3(65536,1,1);
    daxpy_kernel<<<griddim,blockdim>>>(n, a, x, y);
}
```
</div>


# Calling OpenACC-routines from CUDA-programs

- In this scenario we have a (main) program written in CUDA and it calls
  functions written (C/C++/Fortran) with OpenACC extension
    - Interface to these functions must have `extern "C" void func(...)`
- OpenACC routines must relate to CUDA-arrays via `deviceptr`
- The CUDA-codes are still compiled with `nvcc` compiler, e.g.
  `nvcc -c -O4 --restrict -arch=sm_35 call_openacc_from_cuda.cu`
- The OpenACC-codes are compiled again with PGI-compiler e.g.
  `pgcc -c -acc -O4 daxpy_openacc.c`
- Linking must still be done with PGI using `-acc -Mcuda` e.g.
  `pgcc -acc -Mcuda call_openacc_from_cuda.o daxpy_openacc.o`


# Data clauses: deviceptr

- Declare data to reside on the device `deviceptr(var-list)`
    - **on entry/on exit:** do not allocate or move data
- Can be used in `data`, `kernels` or `parallel` constructs
- Heavy restrictions for variables declared in `var-list`:
    - C/C++: Variables must be pointer variables
    - Fortran: Variables must be dummy arguments and may not have the
      `POINTER`, `ALLOCATABLE` or `SAVE` attributes
- In C/C++ (and partially in Fortran) device pointers can be manipulated
  through OpenACC API


# Calling OpenACC-routines from CUDA-programs

<div class="column">
```c
// call_openacc_from_cuda.cu
#include <stdio.h>
#include <cuda.h>
extern "C" void daxpy(int n, double a,
                      const double *x, double *y);
extern "C" void init(int n, double scaling, double *v);
extern "C" void sum(int n, const double *v, double *res);
int main(int argc, char *argv[])
{
    int n = (argc > 1) ? atoi(argv[1]) : (1 << 27);
    double *x, *y, *s, tmp;
    // Allocate data on device
    cudaMalloc((void **)&x, (size_t)n*sizeof(*x));
    cudaMalloc((void **)&y, (size_t)n*sizeof(*y));
    cudaMalloc((void **)&s, (size_t)1*sizeof(*s)); // "sum"
    // All these are written in C + OpenACC
    init(n,  1.0, x); init(n, -1.0, y);
    daxpy(n, a, x, y);
    sum(n, y, s); // A check-sum : "sum(y[0:n])”

    cudaMemcpy(&tmp,s,(size_t)1*sizeof(*s),
                       cudaMemcpyDeviceToHost); // chksum
    cudaFree(x); cudaFree(y); cudaFree(s);
```
</div>

<div class="column">
```c
// daxpy_openacc.c
void daxpy(int n, double a,
           const double *restrict x, double *restrict y)
{
    #pragma acc parallel loop deviceptr(x,y)
    for (int j=0; j<n; ++j)
        y[j] += a*x[j];
}
void init(int n, double scaling, double *v)
{
    #pragma acc parallel loop deviceptr(v)
    for (int j=0; j<n; ++j)
        v[j] = scaling * j;
}
void sum(int n, const double *v, double *res)
{
    #pragma acc kernels deviceptr(v,res)
    {
        double s = 0;
        #pragma acc loop reduction(+:s)
        for (int j=0; j<n; ++j)
            s += v[j];
        // *res = s; // not supported for deviceptr
        #pragma acc loop seq
        for (int j=0; j<1; ++j)
            res[j] = s;
    }
}
```
</div>


# `acc_map_data()` function

- `acc_map_data(hostptr, deviceptr, length)`
    - Instructs the OpenACC runtime that it should map an existing device
      array to the host array.
    - Typically these have been allocated separately with malloc &
      cudaMalloc
- `acc_unmap_data(hostptr)`
    - Reverse
- Functions are only specified for C and C++
- Enables one to use OpenACC functions in Cuda without deviceptr()


# Summary

- Routine directive
    - Enables one to write device functions that can be called within
      parallel loops
- Asynchronous execution
    - Enables better performance by overlapping different operations
- Interoperability
