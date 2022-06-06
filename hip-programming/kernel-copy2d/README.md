# Kernel: copy2d

Write a device kernel that performs the double precision BLAS operation
**dcopy**, i.e. `y = x` using GPU threads in a 2D grid.

Assume that the vectors are used to store a 600x400 matrix (in row-major
format). Initialise the matrix with some values on the CPU and confirm the
correctness of the calculation, e.g. by comparing to reference values
calculated on the CPU or by printing out some of the results.

You may start from a skeleton code provided in [copy2d.cpp](copy2d.cpp).
