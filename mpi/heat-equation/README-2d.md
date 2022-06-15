## Heat equation decomposed in two dimensions

Note: this is very advanced exercise. It is meant more for illustration what can be done
with derived datatypes, and as something one might want to study after the course, rather than
completing during the course.

If you are not familiar with the two dimensional heat equation, please have a look
for [basic description](https://github.com/csc-training/mpi-introduction/tree/main/heat-equation)
in "Parallel programming with MPI" exercise material.

Here, starting point is a working code parallelized over columns (in Fortran) or rows (in C/C++).

Before starting with this exercise, it is recommended that you have
the [usage of Cartesian communicator](README_cartesian.md) implemented.

1. Modify the creation of Cartesian communicator so that the
   decomposition is done in two dimensions, and determine all four
   neighbors (up, down, left, right).
   
2. As the rows (in Fortran) or columns (in C/C++) are not contiguous
   in the computer memory, one needs to use user-defined datatypes
   when communicating in the `exchange()` routine in [cpp/core.cpp](cpp/core.cpp) 
   or [fortran/core.F90](fortran/core.F90). In order to make code more
   symmetric, one can utilize derived type also for the contiguous
   dimension. Create required datatypes (it is recommended to store
   them as attributes in "parallel" data structure).
3. Perform the halo exchange with `MPI_Neighbor_alltoallw`. Together
   with the user defined datatypes, no temporary buffers are needed in
   the user code. In order to use `MPI_Neighbor_alltoallw`, you need
   to determine the correct `displacements` both in sending and
   receiving.
4. In the base version, the I/O routines `write_field` and
   `read_field` (in [cpp/core.cpp](cpp/io.cpp) or
   [fortran/core.F90](fortran/io.F90))
   use temporary buffers for communication. Create appropriate
   datatype and utilize it in I/O related communication. Note that you
   need also the coordinates of processes in the cartesian grid in
   order to read from / write to the correct part of the global
   temperature field.
