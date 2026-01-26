<!--
SPDX-FileCopyrightText: 2010 CSC - IT Center for Science Ltd. <www.csc.fi>

SPDX-License-Identifier: CC-BY-4.0
-->

# Discussion

*Note for instructors: use `run*.sh` scripts to generate all the output files.*


## Task: Compiling and running

1. Compilation succeeds.

2. Outputs are in ...

3. Outputs are in ...

4. ...


## Task: Library functions

1. See `hello-1.cpp`.

2. See `hello-2.cpp`.

   Note that each thread calls `omp_get_num_threads()` in this code.
   This is unnecessary and we'll see later how to execute this function call only with a single thread.

## Bonus task: Conditional compilation

1. The linking fails to undefined reference to `omp_get_thread_num()` and `omp_get_num_threads()`.

2. See `hello-3.cpp` that includes `omp.h` header only if `_OPENMP` is defined and declares dummy functions otherwise.
