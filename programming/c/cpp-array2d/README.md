## Multidimensional array class for heat equation ##

a) Create a two-dimensional `258x258` array of double precision numbers using the `Matrix<T>` class template, provided to you in [matrix.h](matrix.h). Introduce an initialization function that will initialize the input array such that values are:

- `20.0` on the left boundary
- `85.0` on the upper boundary
- `70.0` on the right boundary
- `5.0`  on the lower boundary

and otherwise zeros. A skeleton code is provided in the file [array2d.cpp](array2d.cpp).

b) Write a function that implements a double-for loop for evaluating the Laplacian using the finite-difference approximation:

![](http://quicklatex.com/cache3/2d/ql_b0e3418f715db7b7865384d6ebd6a42d_l3.png)

As an input use the `258x258` array of Exercise a. Evaluate the Laplacian only at the inner `256x256` points, the outer points are used as a boundary condition. As a grid spacing, use `dx=dy=0.01`.

**BONUS**:
c) Generalize the given `Matrix<T>` template class into a temperature field class template `Field<T>`. The class has the following attributes:

- number of grid points `nx` and `ny` in the x- and y-direction
- the grid spacings `dx` and `dy` in the x- and y-direction
- two dimensional data array containing the data points of the field. The array should also contain the boundary values, so its dimensions are `nx+2` and `ny+2`. 

Finally, implement the initialization of the two-dimensional array (Exercise a) and finite-difference Laplacian (Exercise b) in their own functions, which take as input the class representing the temperature field (Exercise c).




