## Hello World with MPI

1. Write a simple parallel program that prints out something (e.g. "Hello
   world!") from multiple processes. Include the MPI headers (C) or use the
   MPI module (Fortran) and call appropriate initialisation and finalisation
   routines.

2. Modify the program so that each process prints out also its rank and so
   that the process with rank 0 prints out the total number of MPI processes
   as well.

3. (Bonus) Sometimes it is convenient to find out in which nodes the different MPI
   processes are running. MPI has a function `MPI_Get_processor_name` for quering the
   name of the node. Use the function in your program (you can search for the interface
   or check the MPI reference in the course material or man pages `man MPI_Get_processor_name`)
   and print out the node name in all processes.
   You can request specific number of nodes with the `--nodes` Slurm option,
   investigate how the results look with different values.
