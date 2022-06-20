## Numerical solution of Laplace equation

The files [jacobi.c](jacobi.c) / [jacobi.F90](jacobi.F90) solve the
the dimensional Laplace equation numerically with the so called Jacobi
iteration. The key component of the numerical algorithm is stencil update,
which is common operation in numerical methods.

Try the offload the stencil updates with appropriate OpenMP constructs.
Compare the performance when code is run on GPU nodes vs. CPU nodes.


