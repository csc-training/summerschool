# GPU programming exercises

There are two types of exercises in this directory.
First you should go over the [fundamentals](exercises/fundamentals) of GPU programming.

After that, in the [bonus](exercises/bonus) exercises you need to apply the fundamental
concepts to solve some more advanced cases.

## Introduction to GPUs

- [Mental model](exercises/fundamentals/01-introduction)

## Basics of HIP programming

### Fundamentals

- [Hello API](exercises/fundamentals/02-kernels/README.md#exercise-compiling-hip-code)
- [Kernel launch](exercises/fundamentals/02-kernels/README.md#exercise-launching-a-kernel)
- [Kernel launch wrapper](exercises/fundamentals/02-kernels/README.md#exercise-error-reporting-from-kernel-launch)
- [API queries](exercises/fundamentals/02-kernels/README.md#exercise-better-error-reporting-by-querying-limits)
- [API errors](exercises/fundamentals/02-kernels/README.md#exercise-errors-from-api-calls)
- [Fill](exercises/fundamentals/02-kernels/README.md#exercise-kernel-for-filling-a-1d-array-with-a-value)
- [Taylor for](exercises/fundamentals/02-kernels/README.md#exercise-re-use-threads-in-a-1d-kernel-with-a-for-loop)

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

## Multi-GPU programming

### Fundamentals

- [Lorem ipsum](exercises/fundamentals/lorem)

### Bonus exercises

- [Lorem ipsum](exercises/bonus/lorem)
