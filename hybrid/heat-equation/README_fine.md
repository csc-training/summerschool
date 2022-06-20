## Heat equation solver in parallel with OpenMP

Parallelise our implementation of a two-dimensional heat equation solver using
OpenMP. See [Code description](code-description.md) for some theory and more
details about the code.

Starting point is a working serial code, which you should parallelize
by inserting appropriate OpenMP directives and routines. For
fine-grain parallelization here, parallel regions should be within
subroutines performing computations.

  1. Determine the number of threads in the main routine ([cpp/main.cpp](cpp/main.cpp) or [fortran/main.F90](fortran/main.F90))
  2. Parallelize the generation of initial temperature in the routine  `generate_field()` (in [fortran/setup.F90](fortran/setup.F90)) or in the `generate()` method (in [cpp/heat.cpp](cpp/heat.cpp)
  3. Parallelize the main computational routine
     `evolve()` in [cpp/core.cpp](cpp/core.cpp) or [fortran/core.F90](fortran/core.F90). 

To build the code, please use the provided `Makefile` (by typing `make`). By default, GNU
compiler is used, in order to use Intel compiler type `make COMP=intel`.
