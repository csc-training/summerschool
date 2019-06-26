## Heat equation solver with MPI+OpenACC

Create a hybrid MPI+OpenACC version of the heat equation solver.

Starting from a pure-MPI parallel version, add parallelisation to the loops of
the time evolution (`evolve()`) using OpenACC.


## Modules to load
1. `pgi/19.1`   
2. `openmpi/3.1.4`   
3. `libpng/1.6`
