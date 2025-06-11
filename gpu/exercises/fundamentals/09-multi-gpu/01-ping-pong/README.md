# Ping-pong with multiple GPUs and MPI

Implement a simple ping-pong test for GPU-to-GPU communication using:
a) indirect communication via the host, and b) direct communication with
HIP-aware MPI.

The ping-pong test constists of the following steps:
  1. Send a vector from one GPU to another
  2. The receiving GPU should increment all elements of the vector by one
  3. Send the vector back to the original GPU


NOTE: Remember to request 2 MPI processes and 2 GPUs when running this exercise. 


```
srun --account=XXXXXX --partition=dev-g -N1 -tasks-per-node=2 --cpus-per-task=1 --gpus-per-node=2 --time=00:15:00 ./a.out
```


## Case 1 - HIP

Start from the skeleton code ([ping-pong.cpp](ping-pong.cpp)). A CPU-to-CPU is already included for reference and as well timing of all tests to compare the execution times.

On **Lumi**, one can compile the MPI example simply using the Cray compiler with
```
CC -xhip ping-pong.cpp
```
and run as indicated above.

## Case 2 - OpenMP Offloading
Based on the solution above write an equivalent code using OpenMP offloading. Replace the memory allocation `hipMalloc((void **) &dA, sizeof(double) * N);` with target data region `#pragma omp target enter data map(alloc: hA[0:N])` and replace `hipmemcpy`  with  `#pragma omp target update` to copy data to and from device.

Direct GPU-to-GPU communications requires a GPU-aware MPI wich uses GPU pointers as arguments. In OpenMP offloading one can require specific calls done by the host to use the GPU pointer of the data by using `use_device_ptr` directive
```
#pragma omp target data use_device_ptr(A)
{
    MPI_Recv(A, N, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
``` 

Finally the kernels are replaced by for loops and `target teams distribute parallel for` directives.
## Case 3 - SYCL  & USM
When using USM with device pinned memory (`malloc_device()`) the SYCL code follows almost one-to-one the HIP code structure. Though there are a few difference to consider. All operations are subumitted to a queue which can be out-of-order (default) or in-order, memory allocations are always blocking, and the `.memcpy()` methods are always non-blocking.
