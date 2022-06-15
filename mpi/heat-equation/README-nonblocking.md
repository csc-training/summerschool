## Non-blocking communication in Heat equation solver

Utilize non-blocking communication in the "halo exchange" of the heat equation solver.
The aim is to be able to overlap the communication and communication. In order to achieve this,
you need to divide the communication and computation into four steps:

1. Initiate the communication in the halo exchange
2. Compute the inner values of the temperature field (those that do not depend on the ghost layers)
3. Finalize the communication in halo exchange
4. Compute the edge values of the temperature field (those that depend on the ghost layers)

Implement the required routines in [fortran/core.F90](fortran/core.F90) or [cpp/core.cpp](cpp/core.cpp), and replace the calls to `exchange` and `evolve` in the `main` routine by the newly 
implemented ones.

In principle it is enough to have the steps 1 and 2 in the [base exercise](README.md) completed
for this exercise, however, we recommend that you have fully working parallel code to start with.


