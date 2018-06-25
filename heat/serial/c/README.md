## Heat equation

Finalize the implementation of the two-dimensional heat equation. You can utilize code written in the previous exercises or use the provided skeleton codes under [heat/c/serial](.). The source files and parts where you need to write code are indicated with TODOs (try e.g. `grep TODO *.c *.h`). There is a ready to use `Makefile`, so you can build the whole code just with `make`.

First, you need two temperature fields for the current and for the previous time steps. Allocate the 2D arrays either based on default dimensions or based on command line arguments. Initialize the boundary values of temperature fields as in Exercise 5a, or read the initial data from `bottle.dat`. Write a function that evaluates the new temperature based on previous one, utilizing the Laplacian:

<!-- Equation
u^{m+1}(i,j) = u^m(i,j) + \Delta t \alpha \nabla^2 u^m(i,j)
-->
![img](http://quicklatex.com/cache3/9e/ql_9eb7ce5f3d5eccd6cfc1ff5638bf199e_l3.png)

The skeleton code `main.c` contains proper values for time step Δt and for the diffusion constant α. Within the main routine evolve the temperature field e.g. for 500 steps and write out every 10th field into a `.png` file.

The program can be run with different command line arguments:

- `./heat` (no arguments - the program will run with the default arguments: 256x256 grid and 500 time steps)
- `./heat bottle.dat` (one argument - start from a temperature grid provided in the file `bottle.dat` for the default number of time steps)
- `./heat bottle.dat 1000` (two arguments - will run the program starting from a temperature grid provided in the file `bottle.dat` for 1000 time steps)
- `./heat 1024 2048 1000` (three arguments - will run the program in a 1024x2048 grid for 1000 time steps)

Visualize the results using `eog` or `animate`: `eog heat_*png` or `animate heat_*png` 

*Note*: model solution contains also a disc in the center of grid as default initial pattern for the temperature field
