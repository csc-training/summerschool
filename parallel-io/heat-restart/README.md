## Heat equation: checkpoint + restart with HDF5

1. Add a feature to the heat equation solver program that enables one to
   start the program from a given state (i.e. not from scratch every time).

2. Add also a checkpointing feature that will dump the state of the simulation
   to disk periodically (e.g. after every few tens of iterations); in a form
   that can be read in afterwards.

Combining the two features allows one to restart and continue an earlier
calculation. So, when the program starts and a checkpoint file is present, it
will replace the initial state with the one in the restart file.

Use MPI-IO to accomplish the I/O routines. Starting points are provided in
[c/io.c](c/io.c).
