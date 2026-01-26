<!--
SPDX-FileCopyrightText: 2010 CSC - IT Center for Science Ltd. <www.csc.fi>

SPDX-License-Identifier: CC-BY-4.0
-->

# OpenMP module

## Learning outcomes

After completing the module, participants should be able to:
- Explain the OpenMP shared‑memory model and its use in modern supercomputers
- Develop parallel applications using OpenMP constructs for loop- and task-based workloads
- Diagnose and fix correctness issues in threaded codes
- Profile and analyze performance of threaded applications
- Run hybrid MPI+OpenMP applications efficiently on modern supercomputers, taking into account process and thread affinities and non-uniform memory access

## Presentation slides

Presentation slides are available [here](https://csc-training.github.io/summerschool/?open=openmp).

## Demos

See [demos directory](demos/) for the demo codes referred to in the slides.

## Exercises

### Getting started

- [Hello world](01-hello/)
- [Calculating axpy](02-axpy/)

### Data sharing

- [Data sharing and parallel regions](03-data/)
note: default is shared, but the loop index is private by default
See https://www.openmp.org/wp-content/uploads/openmp-examples-5.2.2-final.pdf page 15

### Reduction

- [Reduction](03-reduction/); merge two below
  - [Race condition in parallel sum](race-condition/)
  - [Reduction](reduction/)

### Execution control

- [Execution controls](execution-controls/)

TODO: execution control add exercise to add `#pragma omp single` to hello solution to get omp num threads only once.

### OpenMP tasks

- [Simple tasking](simple-tasks/)
- [Parallel Fibonacci](fibonacci/)
- [Parallelizing Mandelbrot with tasks](mandelbrot/)

### Hybrid MPI+OpenMP programming

- [Hybrid Hello World](hybrid-hello/)
- [(Bonus) Multiple thread communication](multiple-thread-communication/)
- [Hybrid heat equation](heat-hybrid)

## Web resources

- OpenMP specifications
  - <https://www.openmp.org/specifications/>
  - See in particular the examples documents that provide many practical examples of use; the API specification is rather technical
