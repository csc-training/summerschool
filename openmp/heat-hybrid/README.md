## Heat equation solver with hybrid MPI + OpenMP

In this exercise, you should parallelize the two dimensional heat equation with hybrid
MPI + OpenMP parallelization. Starting point is a working MPI code parallelized over columns (in Fortran) or rows (in C/C++). See [Code description](code-description.md) for some theory and more
details about the code.

### Fine-grained OpenMP parallelization

Here, parallel regions are only within the computational routines and all the MPI communication 
is handled by the master process. Your tasks are thus:

1. Initialize MPI with appropriate thread safety level
2. Determine the number of threads in the main routine ([cpp/main.cpp](cpp/main.cpp) or [fortran/main.F90](fortran/main.F90))
3. Parallelize the generation of initial temperature in the routine  `generate_field()` (in [fortran/setup.F90](fortran/setup.F90)) or in the `generate()` method (in [cpp/heat.cpp](cpp/heat.cpp)
4. Parallelize the main computational routine
   `evolve()` in [cpp/core.cpp](cpp/core.cpp) or [fortran/core.F90](fortran/core.F90)

### Coarse-grained OpenMP parallelization

Now, there is only one parallel region (within the main routine), however, MPI communication 
in the halo exchange is still done only by one thread (but not necessarily master).

1. Initialize MPI with appropriate thread safety level
2. Insert apprioriate OpenMP directives thoughtout the code

To build the code, please use the provided `Makefile` (by typing `make`). By default, GNU 
compiler is used, in order to use Intel compiler type `make COMP=intel`.
