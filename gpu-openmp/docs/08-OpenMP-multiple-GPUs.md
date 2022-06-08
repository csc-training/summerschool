---
title:  "OpenMP: multi-GPU programming"
author: CSC - IT Center for Science
date:   2021
lang:   en
---


# Multi-GPU programming with OpenMP

![](img/supercomputer-anatomy.png){.center}

<div class=column>
- Three levels of hardware parallelism in a supercomputer:
    1. GPU - different levels of threads
    2. Node - multiple GPUs and CPUs
    3. System - multiple nodes connected with interconnect
</div>

<div class=column>
- Three parallelization methods:
    1. OpenMP offload
    2. OpenMP or MPI
    3. MPI between nodes
</div>


# Multi-GPU communication cases

- Single node multi-GPU programming
    - All GPUs of a node are accessible from a single process and its OpenMP
      threads
    - Data copies either directly or through CPU memory
- Multi-node multi-GPU programming
    - Communication between nodes requires message passing (MPI)
- In this lecture we will discuss in detail only parallelization with MPI
    - It enables direct scalability from single to multi-node


# Multiple GPUs

- OpenMP permits using multiple GPUs within one node by using the
  `omp_get_num_devices` and `omp_set_default_device` functions
- Asynchronous OpenMP calls, OpenMP threads or MPI processes must be used
  in order to actually run kernels in parallel
- Issue when using MPI:
    - If a node has more than one GPU, all processes in the node can
      access all GPUs of the node
    - MPI processes do not have any a priori information about the other
      ranks in the same node
    - Which GPU the MPI process should select?


# Selecting a device with MPI

- Simplest model is to use **one** MPI task per GPU
- Launching job
    - Launch you application so that there are as many MPI tasks per node as
      there are GPUs
    - Make sure the affinity is correct - processes equally split between the
      two sockets (that nodes typically have)
    - Read the user guide of the system for details how to do this!
- In the code a portable and robust solution is to use MPI shared memory
  communicators to split the GPUs between processes
- Note that you can also use OpenMP to utilize all cores in the node for
  computations on CPU side


# Selecting a device with MPI

```c
MPI_Comm shared;
int local_rank, local_size, num_gpus;

MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                    MPI_INFO_NULL, &shared);
MPI_Comm_size(shared, &local_size); // number of ranks in this node
MPI_Comm_rank(shared, &local_rank); // my local rank
num_gpus = omp_get_num_device(); // num of gpus in node
if (num_gpus == local_size) {
    omp_set_default_device(local_rank);
} // otherwise error
```


# Sharing a device between MPI tasks

- In some systems (e.g. with NVIDIA Multi-Process Service) multiple
  MPI tasks (within a node) may share a device efficiently
- Can provide improved performance if single MPI task cannot fully
  utilize the device
- Oversubscribing can lead also to performance degregation,
  performance should always be tested
- Still, only MPI tasks within a node may share devices
- Load balance may be a problem if number of MPI tasks per device is
  not the same for all the devices


# Assigning multiple MPI tasks per device

```c
MPI_Comm shared;
int local_rank, local_size, num_gpus, my_device;

MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                    MPI_INFO_NULL, &shared);
MPI_Comm_size(shared, &local_size); // number of ranks in this node
MPI_Comm_rank(shared, &local_rank); // my local rank
num_gpus = omp_get_num_device(); // num of gpus in node
my_device = local_rank % num_gpus; // round robin assignment with modulo
omp_set_default_device(my_device);
```


# Data transfers

- Idea: use MPI to transfer data between GPUs, use OpenMP offloading for
  computations
- Additional complexity: GPU memory is separate from CPU memory
- GPU-aware MPI-library
    - Can use the device pointer in MPI calls - no need for additional buffers
    - No need for extra buffers and device-host-device copies
    - If enabled on system, data will be transferred via transparent RDMA
- Without GPU-aware MPI-library
    - Data must be transferred from the device memory to the host memory and
      vice versa before performing MPI-calls


# Using device addresses with MPI

- For accessing device addresses of data on the host one can use
  `target data` with the `use_device_ptr` clause
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
#pragma omp target data use_device_ptr(data)
{
    MPI_Send(data, N, MPI_DOUBLE, to, MPI_ANY_TAG, MPI_COMM_WORLD);
}

/* MPI_Recv with GPU-aware MPI */
#pragma omp target data use_device_ptr(data)
{
    MPI_Recv(data, N, MPI_DOUBLE, from, MPI_ANY_TAG, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
}
```


# Summary

- Typical HPC cluster node has several GPUs in each node
    - Selecting the GPUs with correct affinity
- Data transfers using MPI
    - GPU-aware MPI avoids extra memory copies
