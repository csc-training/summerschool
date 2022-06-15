## Heat equation solver in parallel with MPI

Parallelise our implementation of a two-dimensional heat equation solver using
MPI. See [Code description](code-description.md) for some theory and more
details about the code.

To parallelise the code, one needs to divide the grid into blocks of columns
(in Fortran) or rows (in C/C++) and assign each block to one MPI task. Or in other
words, share the work among the MPI tasks by doing a domain decomposition.

![2D domain decomposition](img/domain-decomposition.svg)

The MPI tasks are able to update the grid independently everywhere else than
on the boundaries -- there the communication of a single column (or row) with
the nearest neighbour is needed. This can be achieved by having additional
ghost-layers that contain the boundary data of the neighbouring tasks. As the
system is aperiodic, the outermost ranks communicate with only one neighbour,
and the inner ranks with two neighbours.

Some parts of the code are already parallelized (*e.g.* input/output), complete
the parallelization as follows (marked with TODOs in the source code):

  1. Initialize and finalize MPI in the main routine ([cpp/main.cpp](cpp/main.cpp) or [fortran/main.F90](fortran/main.F90))
  2. Determine the number of MPI processes, rank, as well as the left (or up) and right (or down) neighbours
     of a domain in the routine `parallel_setup()` (in [fortran/heat_mod.F90](fortran/heat_mod.F90)) or in the `ParallelData()` constructor (in [cpp/heat.hpp](cpp/heat.hpp)
  3. Use `MPI_Send` and `MPI_Recv` for implementing the "halo exchange" operation in the 
     `exchange()` routine in [cpp/core.cpp](cpp/core.cpp) or [fortran/core.F90](fortran/core.F90). 

To build the code, please use the provided `Makefile` (by typing `make`). By default, Intel 
compiler is used, in order to use gcc type `make COMP=gnu`.
