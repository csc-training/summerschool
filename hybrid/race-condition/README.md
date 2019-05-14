## Vector sum and race condition ##

In [vectorsum.c](vectorsum.c) (or [vectorsum.F90](vectorsum.F90) for
Fortran) you will find a serial code that sums up all the elements of
a vector A, initialized as A=1,2,...,N. For reference, from the
arithmetic sum formula we know that the result should be S=N(N+1)/2.

Try to parallelize the code by using `omp parallel` or `omp for`
pragmas. Are you able to get same results with different number of
threads and in different runs? Explain why the program does not work
correctly in parallel. What is needed for correct computation?
