## Hello World with MPI

1. Write a parallel program that prints out something (e.g. "Hello
   world!") from multiple processes. Include the MPI headers (C) or use the
   MPI module (Fortran) and call appropriate initialisation and finalisation
   routines.

2. Modify the program so that each process prints out also its rank and so
   that the process with rank 0 prints out the total number of MPI processes
   as well.

3. (Bonus) Sometimes it is convenient to find out in which nodes the different MPI
   processes are running. MPI has a function `MPI_Get_processor_name` for querying the
   name of the node. Use this function in your program and print out the node name in all processes.
   Use any of the following sources to find the usage of `MPI_Get_processor_name`:

   - The official MPI specification: https://www.mpi-forum.org/docs/
   - Manual pages with Linux `man` command: `man MPI_Get_processor_name`
   - Documentation pages of [OpenMPI](https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man3/MPI_Get_processor_name.3.html)
   or [MPICH](https://www.mpich.org/static/docs/v3.2/www3/MPI_Get_processor_name.html), both excellent resources.
   Function signatures are the same in both (all) MPI implementations as these are fixed by the specification.

   When submitting your program to Slurm, you can request a specific number of nodes with the `--nodes` Slurm option.
   Investigate how the results look with different values.
