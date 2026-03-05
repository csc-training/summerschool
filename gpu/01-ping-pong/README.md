# Ping-pong with multiple GPUs and MPI

Implement a ping-pong test for GPU-to-GPU communication indirectly via host and
directly between the GPUs. 

The test constists of the following steps:
  1. Initialize a vector on GPU-1 and send it to GPU-2
  2. Increment values of the vector by one on GPU-2
  3. Send the vector back to GPU-1 from GPU-2

## Notes
- Remember to request 2 MPI processes and 2 GPUs. For example, the following
  reserves one node, 2 tasks on it and 2 gpus on the node. You may experiment
  other configurations as well.
```
srun --account=XXXXXX --partition=dev-g -N1 --tasks-per-node=2 --cpus-per-task=1 --gpus-per-node=2 --time=00:15:00 ./a.out
```
- GPU-aware MPI on LUMI requires setting environment variable `MPICH_GPU_SUPPORT_ENABLED=1`.

## Case 1 - HIP

You may base your solution on the skeleton code
([ping-pong.cpp](ping-pong.cpp)). A CPU-to-CPU is already included for
reference and as well timing of all tests to compare the execution times.

On **Lumi**, one can compile the MPI example simply using the Cray compiler with
```
CC -xhip ping-pong.cpp
```
and run as indicated above.

## Case 2 - OpenMP Offloading

Based on the solution above write an equivalent code using OpenMP offloading.
Replace the memory allocation (`hipMalloc((void **) &dA, sizeof(double) * N);`)
with target data region `#pragma omp target enter data map(alloc: hA[0:N])` and
hip memory copy (`hipMemcpy`) with target update (`#pragma omp target update`).

Direct GPU-to-GPU communications requires a GPU-aware MPI which uses GPU
pointers as arguments. In OpenMP offloading one enforce the use of the device
pointer of an array with `use_device_ptr`:
```
#pragma omp target data use_device_ptr(A)
{
    MPI_Recv(A, N, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
```
Without the pragma, the host pointer of `A` would be used.

Finally the kernels are replaced by for loops and `target teams distribute
parallel for` directives.

On LUMI, the code is compiled with 
```
CC -fopenmp ping-pong.cpp
```
