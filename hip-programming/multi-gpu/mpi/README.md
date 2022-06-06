# Multi-GPU exercise with MPI

The purpose of this exercise is to demonstrate the use of MPI for GPU-GPU communication. The program performs an MPI transfer between two processes using host data and device data. The device data transfer is demonstrated by two methods; by using a manual host staging, and directly passing device pointers to MPI. Fill the blanks indicated by #error.

At AAC cloud, compile with
```
srun --reservation=Lumi --time=00:01:00 -p MI100 --nodes 1 make
```
Run with
```
srun --reservation=Lumi --time=00:01:00 -p MI100 --nodes 1 -n 2  mpirun -np 2 ./mpiexample
```
