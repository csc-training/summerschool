---
title:  "OpenACC: advanced topics"
author: CSC Summerschool
date:   2019-07
lang:   en
---

# Asynchronous operations {.section}


# Motivation

- By default, the local thread will wait until OpenACC compute or data
  construct has completed its execution
- Potential parallelism in overlapping compute, data transfers, MPI,
  etc.

![](img/synchronous.png){.center}

# Asynchronous execution: async and wait

- `async[(int-expr)]` **clause** enables one to enqueue compute and
  data operations, and local (host) thread will continue execution
    - Order is preserved for enqueued operations
    - OpenACC `async` clause is supported by constructs:  
     `parallel`, `kernels`,  
     `enter data`, `exit data`,  
     `update`, `wait`

- `wait[(int-expr-list)]` **directive** causes the CPU thread to wait for completion of asynchronous
  operations
    - C/C++: `#pragma acc wait [(int-expr-list)] [clauses]`
    - Fortran: `!$acc wait [(int-expr-list)] [clauses]`


# OpenACC and asynchronous execution

![](img/async.png){.center}


# Multiple queues

- One can have multiple queues, enabling one to overlap execution of
  kernels and data operations
- `async` clause
    - non-negative integer argument, defining on which queue the operation
      is placed
    - Within one queue order is preserved, in different queues operations
      have no coupling
    - If no queue is given the default queue is used
- `wait` directive
    - list of integers as argument, defining which queues to wait on.
    - By default it waits for all.


# OpenACC and asynchronous execution

![](img/2queues.png){.center}



# Example c = a + b (1/2)
```c
//Initialization of a,b,c

a = malloc(sizeof(double) * N);
b = malloc(sizeof(double) * N);
c = malloc(sizeof(double) * N);

for (int i = 0; i < N;i++) {
    a[i] = i;
    b[i] = i;
}
```
# Example c = a + b (2/2)

```c
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

# Multi-GPU programming with OpenACC {.section}


# Multi-GPU programming with OpenACC

<div class=column>
Three levels of hardware parallelism in a supercomputer

1. GPU - different levels of threads
2. Node - GPU, CPU and interconnect
3. Machine - several nodes connected with interconnect
</div>
<div class=column>
Three parallelization methods

1. OpenACC
2. OpenMP or MPI 
3. MPI between nodes
</div>

![](img/gpu-cluster.png){.center}

# Multi-GPU communication cases

- Single node multi-GPU programming
    - All GPUs of a node are accessible from single process and its OpenMP threads 
    - Data copies either directly or through CPU memory
- Multi node multi-GPU programming
    - Communication between nodes requires message passing, MPI
- In this lecture we will in detail only discuss parallelization with MPI
    - This enables direct scalability from single to multi-node



# Selecting device

- OpenACC permits using multiple GPUs within one node by using the
  `acc_get_num_devices` and `acc_set_device_num` functions
- Asynchronous OpenACC calls, OpenMP threads or MPI processes must be used
  in order to actually run kernels in parallel
- Issue when using MPI:
    - If a node has more than one GPU, all processes in the node can
      access all GPUs of the node
    - MPI processes do not have a priori information on the other ranks in
      the same node
    - Which GPU the MPI process should select?



# Selecting the device in MPI

- Model is to use **one** MPI task per GPU
- Launching job
    - Launch you application so that there are as many MPI tasks per node as there are GPUs
    - Make sure the affinity is correct - processes equally split between the two sockets (that nodes typically have)
    - Read the user guide of the system for details how to do this! 
- In the code a portable and robust solution is to use MPI3 shared memory
    communicators to split the GPUs between processes
- Note that you can also use OpenMP to utilize all cores in the node for computations on CPU side

# Selecting the device in MPI

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


# Data transfers

- Idea: use MPI to transfer data between GPUs, use OpenACC-kernels for
  computations
- Additional complexity: GPU memory is separate from that of a CPU
- GPU-aware MPI-library
    - Can use the device pointer in MPI calls - no need for additional buffers
    - No need for extra buffers and device-host-device copies
    - If enabled on system data will be transferred via transparent RDMA
- Without GPU-aware MPI-library
    - Data must be transferred from the device memory to the host memory and vice versa before performing
      MPI-calls

# Using device addresses with host_data

- For accessing device addresses of data on the host OpenACC includes
  `host_data` construct with the `use_device` clause
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
<small>
Example from
<https://devblogs.nvidia.com/parallelforall/7-powerful-new-features-openacc-2-0/>
</small>
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


# Summary
- Asynchronous execution
    - Enables better performance by overlapping different operations
- Typical HPC cluster node has several GPUs in each node
    - Selecting the GPUs with correct affinity
    - Data transfers using MPI
- Routine directive
    - Enables one to write device functions that can be called within
      parallel loops

