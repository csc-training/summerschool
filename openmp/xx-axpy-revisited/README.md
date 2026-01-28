<!--
SPDX-FileCopyrightText: 2010 CSC - IT Center for Science Ltd. <www.csc.fi>

SPDX-License-Identifier: CC-BY-4.0
-->

# Exercise: Revisiting the performance of axpy

In this exercise we improve the performance of the axpy code by taking into account non-uniform memory access in modern processors.

An example code is provided from the previous axpy exercise: a fully functional axpy code with timing.

The [solution directory](solution/) contains a model solution and discussion on the exercises below.


## Task: Improve the performance

1. The arrays are allocated as `std::vector` that does zero-initialization of the underlying memory buffer on the calling thread.
   According to first touch policy, this means that the whole array gets allocated in the memory of the main thread's NUMA domain.

   Fix this issue by first allocating the memory buffer with `malloc()` and mapping the buffer to a vector-like object `std::span`:

       double *_x = (double*)malloc(n * sizeof(double));
       std::span<double> x(_x, n);  // this x behaves like std::vector for the purposes of this exercise
       ...
       free(_x);

   Then, use OpenMP for the data initialization loop so that the same threads that will do the axpy compute have
   initialized their data.

   Do you see any difference in execution time with different numbers of threads and different affinities?

2. (Bonus) Use C++ unique pointer to wrap the C-style malloc and free for automatic freeing of memory.
