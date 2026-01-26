<!--
SPDX-FileCopyrightText: 2010 CSC - IT Center for Science Ltd. <www.csc.fi>

SPDX-License-Identifier: CC-BY-4.0
-->

# Exercise: Calculating axpy

In this exercise we practise parallelizing an axpy code using OpenMP threading.

The axpy operation is a fundamental linear algebra operation defined as

$$
y_i \leftarrow \alpha x_i + y_i
$$

where $\alpha$ is a scalar and $x$ and $y$ are vectors of the same size.

A serial example code is provided: The code initializes the input values $\alpha$, $x$, and $y$,
performs axpy operation, and prints the output $y$.

The [solution directory](solution/) contains a model solution and discussion on the exercises below.


## Task: Parallelize with OpenMP threads

1. Study, compile, and run the provided code. You can provide the array size as a command line argument, e.g., `./axpy.x 1024`.

   Note 1: the function `run()` in the top of the file is of most interest for us.
   The `main()` function at the bottom of the file does the command line argument parsing that needs no editing.

   Note 2: the code includes a separate `helper_functions.{hpp,F90}` file that provides the `print_array()`
   helper function. You don't need to study the contents of the helper file, but you can use it as such.

2. Parallelize the axpy loop by adding suitable OpenMP directives (see 'TODO' in the code).
   Test that the code gives correct values using different number of threads.

3. (Bonus) Do you see any difference in execution time with threads? Try out different array sizes too.

   Hint: use `omp_get_wtime()` function to measure the time spent in the axpy loop.
   See [OpenMP documentation](https://www.openmp.org/spec-html/5.0/openmpsu160.html#x199-9660003.4.1)
   for example use of the function.
