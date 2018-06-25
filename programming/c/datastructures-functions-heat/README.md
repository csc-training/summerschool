## Data structures and functions for heat equation

A skeleton code for these exercises are provided under this folder.

a) Create a two-dimensional `258x258` array of double precision numbers. Initialize the array such that the values are:

- `20.0` on the left boundary
- `85.0` on the upper boundary
- `70.0` on the right boundary
- `5.0`  on the lower boundary

and otherwise zeros. A skeleton code is provided in [2d_array.c](2d_array.c).


b) Write a double-for loop for evaluating the Laplacian using the finite-difference approximation:

![](http://quicklatex.com/cache3/2d/ql_b0e3418f715db7b7865384d6ebd6a42d_l3.png)

As an input use the `258x258` array of Exercise a (or start from the skeleton [laplacian.c](laplacian.c). Evaluate the Laplacian only at the inner `256x256` points, the outer points are used as a boundary condition. As a grid spacing, use `dx=dy=0.01`.

c) Create a struct for a temperature field. The struct has the following elements:

- number of grid points `nx` and `ny` in the x- and y-direction
- the grid spacings `dx` and `dy` in the x- and y-direction
- the squares of grid spacings `dx2` and `dy2`
- two dimensional array containing the data points of the field. The array contains also the boundary values, so its dimensions are `nx+2` and `ny+2`.

d) Implement the initialization of the two dimensional array (exercise a) and finite-difference Laplacian (exercise b) in their own functions, which take as input the struct representing the temperature field (exercise 5c).
