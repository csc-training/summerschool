## Collective I/O

In this exercise we practice outputting data to disk in MPI setting.
We will implement file output first using standard I/O routines with MPI communication,
and then again using MPI-IO parallel write routines.
The exercise tries to mimic a typical I/O situation in HPC simulations where
each MPI process holds data that we want to store in a file using some deterministic order of writes.
For example, if the data represents values of some simulated quantity on a distributed grid,
we may want to ensure that the data is written in the same order as the grid points are indexed.


Have a look at one of the following unfinished programs:

- [](collective-io.cpp) (`C/C++`)
- TODO fortran

These create an integer array on each MPI rank and initialize its values to rank of the process.
The number of elements in the array (amount of data to be written) can be set through a command line argument.
There are two unfinished functions `single_writer()` and `collective_write`: both of these are intended to be called by all
MPI processes and should produce a **single** file on disk containing the full input data from all ranks.

Your task is to implement these functions as follows.

1. `single_writer()` should write the input data to a single file on disk using the "spokesperson" strategy,
ie. data is collected to MPI rank 0, which then writes it to file using standard library I/O.
2. Using the collective MPI-IO routine `MPI_File_write_at_all()`.

In both cases the writes should be ordered so that data from rank 0 is written first,
then data from rank 1 _etc_.