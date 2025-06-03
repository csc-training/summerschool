<!-- Adapted from material by EPCC https://github.com/EPCCed/archer2-MPI-2020-05-14 -->

# Parallel calculation of π

In [the previous generalized parallel pi exercise](../parallel-pi-general),
the final reduction of partial sums to rank 0 was done using point-to-point send and receive operations.
In this exercise, the task is to improve the code by using an MPI reduction operation instead.

Starting from your solution (or the model solution) to this previous exercise, complete the following tasks:

1. Replace all the `MPI_Send` and `MPI_Recv` calls with a single `MPI_Reduce` call to sum up the partial sums to rank 0.
   Print the result on rank 0.
   Do you get **exactly** the same result as in the previous exercise?

2. Use `MPI_Allreduce` instead to make all ranks have the final sum.

3. Use `MPI_IN_PLACE` in the `MPI_Allreduce` call to avoid creating an extra variable for the final sum.

