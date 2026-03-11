# Vector sum on two GPUs without MPI

Calculate the vector sum of two vectors (C = A + B) using two GPUs.

NOTE: Remember to request 2 GPUs on single node when running this exercise. On Lumi, use
```
srun --account=XXXXXX --partition=dev-g -N1 -n1 --cpus-per-task=1 --gpus-per-node=2 --time=00:15:00 ./a.out
```

## Case 1 - HIP

Start from the skeleton code [vector-sum.cpp](vector-sum.cpp). Decompose the
vectors into equal halves, copy data from host to device memory and launch a
GPU kernel on each part asynchronously using streams. Copy the results back to
the host to check for correctness. Add timing events to measure the time of
execution.

Assuming all modules are loaded corrected, On **Lumi**, one can compile the MPI
example simply using the Cray compiler with
```
CC -fopenmp ping-pong.cpp
```
and run as indicated above.

## Case 2 - OpenMP Offloading

Similar behaviour can be modeled using OpenMP offloading. OpenMP target tasks
can be launched in parallel if `nowait` clause is used. Each has to use a
different device which can be specified with `device(i)` clause. Finally each
task will map and work on only half of the data. After the tasks are launched
one can pause the program until all work is done using `#pragma omp taskwait` 
