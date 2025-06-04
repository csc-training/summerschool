# Two dimensional heat equation with hybrid parallelization

This directory contains a working implementation of two dimensional
heat equation both in C++ and in Fortran. Code has been parallelized in a
hybrid fashion with MPI and OpenMP.

1. Read a short [description](code-description.md) of the theory and the
   numerical implementation in the code.
2. Get yourself familiar with the code and basic ideas behind the
   parallelization.
3. Build the code with the provided Makefile (by typing `make`).
4. Try to run the code in Mahti with different parallelization options, and
   investigate the performance. You may try also different input parameters.
   There is a template batch job script [job_hybrid.sh](job_hybrid.sh) that
   you can use for hybrid MPI+OpenMP runs.
