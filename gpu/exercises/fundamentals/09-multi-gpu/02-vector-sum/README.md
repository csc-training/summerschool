# Vector sum on two GPUs without MPI

Calculate the vector sum of two vectors (C = A + B) using two GPUs.

NOTE: Remember to request 2 GPUs when running this exercise. On Lumi, use
```
srun --account=XXXXXX --partition=dev-g -N1 -n1 --cpus-per-task=1 --gpus-per-node=2 --time=00:15:00 ./a.out
```

## Case 1 - HIP
Start from the skeleton code [vector-sum.cpp](vector-sum.cpp).
Decompose the vectors into equal halves, copy data from host to device memory
and launch a GPU kernel on each part asynchronously using streams. Copy the
results back to the host to check for correctness. Add timing events to
measure the time of execution.

Assumin all modules are loaded corrected, 
On **Lumi**, one can compile the MPI example simply using the Cray compiler with
```
CC -fopenmp ping-pong.cpp
```
and run as indicated above.

## Case 2 - openMP Offloading
Similar behaivour can be modeled using OpenMP offloading. OpenMP taget tasks can be launched in parallel if `nowait` clause is used. Each has to use a different device which cen be specify usingg `device(i)` close and finally each task will map and work on only half of the data. After the tasks are launched one can pause the program untill all work is done using `#pragma omp taskwait` 
## Case 3 - SYCL & USM
hen using USM with device pinned memory (`malloc_device()`) the SYCL code follows almost one-to-one the HIP code structure. Each device has its own queue and all operations except the memory allocations are asynchronous. After all the work is submit to the devices the host will wait for their completion using the queue method `.wait()` or sycl events.