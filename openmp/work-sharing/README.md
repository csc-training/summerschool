## Work sharing for a simple loop ##

In [sum.c](sum.c) (or [sum.F90](sum.F90) for Fortran) is a skeleton
implementation for a simple summation of two vectors, C=A+B. Add the
computation loop and add the parallel region with work sharing
directives so that the vector addition is executed in parallel.
