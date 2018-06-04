## Hybrid heat equation solver revisited

Continue work on your [hybrid implementation](../heat-fine) of the heat
equation solver.

Now combine the OpenMP implementation, where the threads are kept alive
throughout the program execution, with the same MPI version. Now the MPI
communication in the halo exchange is carried out in the
**MPI_THREAD_SERIALIZED** or **MPI_THREAD_MULTIPLE** mode.
