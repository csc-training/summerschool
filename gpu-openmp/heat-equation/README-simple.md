## Heat equation solver in parallel with OpenMP

Parallelise our implementation of a two-dimensional heat equation solver
using OpenMP offloading. See [Code description](code-description.md) for some
theory and more details about the code.

Starting point is a working serial code, which you should parallelize
by inserting appropriate OpenMP directives and routines.

Try to accelerate the code by adding basic OpenMP offload constructs into the
main computational routine `evolve()` in [cpp/core.cpp](cpp/core.cpp) or
[fortran/core.F90](fortran/core.F90).

Compare the performance of CPU only and GPU accelerated versions.

To build the code, please use the provided `Makefile` (by typing `make`).
