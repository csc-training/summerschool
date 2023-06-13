# Ping-pong with multiple GPUs and MPI

Implement a simple ping-pong test for GPU-to-GPU communication using:
a) indirect communication via the host, and b) direct communication with
HIP-aware MPI.

The ping-pong test constists of the following steps:
  1. Send a vector from one GPU to another
  2. The receiving GPU should increment all elements of the vector by one
  3. Send the vector back to the original GPU

For reference, there is also a CPU-to-CPU implementation in the skeleton
code ([ping-pong.cpp](ping-pong.cpp)). Timing of all tests is also included to
compare the execution times.


NOTE: Remember to request 2 MPI processes and 2 GPUs when running this exercise. 

On **Lumi**, one can compile the MPI example simply using the Cray compiler with
```
CC -xhip ping-pong.cpp
```
and run with
```
srun --account=XXXXXX --partition=dev-g -N1 -n2 --cpus-per-task=1 --gpus-per-task=2 --time=00:15:00 ./a.out
```

On **Puhti**, compile the MPI example with
```
OMPI_CXXFLAGS='' OMPI_CXX='hipcc --x cu' mpicxx -c -o ping-pong.o ping-pong.cpp
```
then link with
```
hipcc ping-pong.o -lmpi
```
and finally run:
```
srun --account=XXXXXX --partition=gputest -N1 -n2 --cpus-per-task=1 --gres=gpu:v100:2 --time=00:15:00 ./a.out
```