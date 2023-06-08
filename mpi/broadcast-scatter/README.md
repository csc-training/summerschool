<!-- Adapted from material by EPCC https://github.com/EPCCed/archer2-MPI-2020-05-14 -->

# Broadcast and scatter

Write a program using send and receive calls to implement **broadcast** and **scatter**
operations. We will see later in the course how to do this using MPI’s collective operations,
but it is useful to implement them by hand as they illustrate how to communicate all or part
of an array of data using point-to-point operations.

1. Create an integer array of size N (e.g. N=12) and initialize it as follows:
   - rank 0: i (i=1,12)
   - other ranks: -1
2. Print out the initial values in all the processes
3. Implement **broadcast**: rank 0 sends the contents of the entire array to other ranks.
   Print out the final values in all the processes.
4. Implement **scatter**: imagine that the array is divided into blocks of size N / P (P=number of
   processes), and rank 0 sends a different block to other ranks, so that the rank *i* receives
   the *i*th block. Print out the final values in all the processes.


You may start from scratch or use [skeleton.cpp](c/skeleton.cpp) (or
[skeleton.F90](fortran/skeleton.F90) for Fortran) as a starting point.

