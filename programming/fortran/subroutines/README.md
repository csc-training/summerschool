## Subroutines and modules

Write three subroutines for i) array initialization, ii) computing a
Laplacian, iii) printing out a given array, and put them in a separate module.
Then, write a main program that defines two (dynamically allocatable) arrays,
`previous` and `current`; initialize the `previous` array; apply the 
Laplacian on the initialized array; and print out both arrays 
(after and before the Laplacian) by calling the three module procedures
described above. You can use the provided skeleton codes
[laplacian.F90](laplacian.F90) and [main.F90](main.F90)
