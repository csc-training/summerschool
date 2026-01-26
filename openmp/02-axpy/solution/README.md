<!--
SPDX-FileCopyrightText: 2010 CSC - IT Center for Science Ltd. <www.csc.fi>

SPDX-License-Identifier: CC-BY-4.0
-->

# Discussion

## Task: Parallelize with OpenMP threads

1. The serial code can be compiled and run as follows:

       g++ -O3 axpy.cpp -o axpy.x
       ./axpy.x 1024

   Output:

       Array size n = 1024
       Input:
       a =   3.0000
       x =   0.0000   0.0010   0.0020   0.0029 ...   0.9971   0.9980   0.9990   1.0000
       y =   0.0000   0.0978   0.1955   0.2933 ...  99.7067  99.8045  99.9022 100.0000
       Output:
       y =   0.0000   0.1007   0.2014   0.3021 ... 102.6979 102.7986 102.8993 103.0000

2. See `axpy.cpp`. Compiling and running:

       g++ -O3 -fopenmp axpy.cpp -o axpy.x
       OMP_NUM_THREADS=2 ./axpy.x 1024

   Output:

       Array size n = 1024
       Input:
       a =   3.0000
       x =   0.0000   0.0010   0.0020   0.0029 ...   0.9971   0.9980   0.9990   1.0000
       y =   0.0000   0.0978   0.1955   0.2933 ...  99.7067  99.8045  99.9022 100.0000
       Output:
       y =   0.0000   0.1007   0.2014   0.3021 ... 102.6979 102.7986 102.8993 103.0000

3. See `axpy-timed.cpp`.

   Calculating axpy of a large array of 102400000 elements takes on Mahti about 78 ms with 1 thread.
   The fastest execution we can get is about 62 ms with 2 or more threads.

   This seems rather appalling performance improvement for such a code with independent loop iterations.
   The reason is two-fold: 1) the axpy operation is bound by the memory bandwidth and 2) the current
   code is not managing memory allocation well for high performance.

   We'll resolve this issue later when we discuss non-uniform memory access on modern processors
   (see the follow-up exercise).
