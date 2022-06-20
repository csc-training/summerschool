## Using data environments: vector sum and dot product

Calculate first a vector sum `C = A + B` and then the dot product of two vectors 
(`r = sum[ C(i) x B(i) ]`) in parallel using OpenMP.

A working serial code is provided in `sum-dot(.c|.F90)`.

Perform the two computations with separate `target` constructs, but utilize `target data` 
for creating a data environment and avoiding unnecessary memory copies between computations.

