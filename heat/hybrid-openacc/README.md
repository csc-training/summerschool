## Heat equation solver with MPI+OpenACC

Create a hybrid MPI+OpenACC version of the heat equation solver.

Starting from a pure-MPI parallel version, add parallelisation to the loops of
the time evolution (`evolve()`) using OpenACC.
