## Heat equation
Finalize the implementation of our two-dimensional heat equation solver
(see the Appendix) by filling in the missing pieces of code
(marked with “TODO” in files [core.F90](core.F90) and [io.F90](io.F90)).
You can compile the code with the provided makefile.

a) The main task is to write the procedure that evaluates the temperature field
based on a previous iteration (the routine is called `evolve` here):

<!-- Equation
u^{m+1}(i,j) = u^m(i,j) + \Delta t \alpha \nabla^2 u^m(i,j)
-->
![img](http://quicklatex.com/cache3/9e/ql_9eb7ce5f3d5eccd6cfc1ff5638bf199e_l3.png)

utilizing our earlier implementation of the Laplacian in the last term.
The skeleton codes readily contain suitable values for time step and for 
the diffusion constant. Run the code with the default initialization.

b) Another task is to implement a reading in of the initial field from a file, 
routine `read_field` located in the [file io.F90](file io.F90). Test your 
implementation with the provided [bottle.dat](bottle.dat).
The complete solver is provided in [solution](solution).

