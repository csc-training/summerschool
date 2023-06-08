## Heat equation solver with Cartesian communicator

If you are not familiar with the two dimensional heat equation, please have a look
for [basic description](https://github.com/csc-training/mpi-introduction/tree/main/heat-equation)
in "Parallel programming with MPI" exercise material.

Here, starting point is a working code parallelized over columns (in Fortran) or rows (in C/C++).

The current version uses only MPI_COMM_WORLD, and neighboring process are determined manually.

1. Add a "communicator" attribute to the basic parallelization data structure (`type :: parallel_data` in [fortran/heat_mod.F90](fortran/heat_mod.F90) or class `ParallelData` in [cpp/heat.hpp](cpp/heat.hpp))
2. Create the Cartesian communicator in the routine `parallel_setup()` (Fortran) or in the
the `ParallelData()` constructor (C++), and use `MPI_Cart_shift` for determining the
neighboring processes
3. Use the Cartesian communicator in all communication routines

To build the code, please use the provided `Makefile` (by typing `make`). By default, Intel
compiler is used, in order to use gcc type `make COMP=gnu`.
