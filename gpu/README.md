# GPU programming exercises

There are two types of exercises in this directory.
First you should go over the [fundamentals](exercises/fundamentals) of GPU programming.

After that, in the [bonus](exercises/bonus) exercises you need to apply the fundamental
concepts to solve some more advanced cases.

## Introduction to GPUs

- [Mental model](exercises/fundamentals/01-introduction)

## Basics of HIP programming

### Fundamentals

- [Seven exercises on the basics](https://github.com/csc-training/summerschool/blob/master/gpu/exercises/fundamentals/02-kernels/README.md)

### Bonus exercises

- [Debugging & fixing errors](exercises/bonus/02-kernels/kernel_errors)
- [Copy 2D kernel](exercises/bonus/02-kernels/copy2d)

## Streams, events, and synchronization

### Fundamentals

- [Four exercise on stream and events](exercises/fundamentals/03-streams)

## Memory management

### Fundamentals

- [API learning exercise](exercises/fundamentals/04-memory)

## Kernel optimizations

### Fundamentals

- [Investigate coalescing](exercises/fundamentals/06-optimization/01-coalescing)
- [Tracing with rocprof](exercises/fundamentals/06-optimization/02-trace)

### Bonus exercises

- [Matrix transpose](exercises/bonus/06-optimization/matrix-transpose)

## OpenMP offloading

### Fundamentals

- [Hello world with OpenMP offloading](exercises/fundamentals/07-openmp/01-hello-world)
- [Using data environments](exercises/fundamentals/07-openmp/02-sum-dot)
- [Data movement clauses and reductions](exercises/fundamentals/07-openmp/03-dot-product)
- [Mimic HIP](exercises/fundamentals/07-openmp/04-mimic-hip)

### Bonus exercises

- [Heat equation](exercises/bonus/07-openmp/heat-equation)
- [Interoperability with HIP](exercises/bonus/07-openmp/hip-interop)
- [Jacobi](exercises/bonus/07-openmp/jacobi)

## SYCL

### Fundamentals

- [SYCL exercises](exercises/fundamentals/08-sycl)

## Multi-GPU programming

### Fundamentals

- [Ping-pong with multiple GPUs and MPI](exercises/fundamentals/09-multi-gpu/01-ping-pong)
- [Vector sum on two GPUs without MPI](exercises/fundamentals/09-multi-gpu/02-vector-sum)
- [Peer to peer device access](exercises/fundamentals/09-multi-gpu/03-p2pcopy)
