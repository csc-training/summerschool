## Cartesian grid process topology

Explore a Cartesian grid process topology by writing a toy program, where

- the MPI processes are arranged into a 2D Cartesian grid
- every task prints out their linear rank together with its coordinates
  in the process grid
- every task prints out the linear rank of their nearest neighboring
  processes

Run the code with both periodic and non-periodic boundaries, and experiment
with the direction and displacement parameters of the `MPI_Cart_shift`
routine.

You can start from scratch or use one of the provided skeleton codes
([skeleton.c](skeleton.c) or [skeleton.f90](skeleton.f90)).
