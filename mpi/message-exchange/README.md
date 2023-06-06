## Message exchange

Write a program where two processes send and receive a message to/from
each other using `MPI_Send` and `MPI_Recv`.

The message content should be an integer array, where each element is
initialised to the rank of the process. After receiving a message, each
process should print out its own rank, number of elements it received,
and the first element in the received array.

You may start from scratch or use [exchange.cpp](exchange.cpp) (or
[exchange.F90](exchange.F90) for Fortran) as a starting point.

Try increasing the message size (e.g. to 100000), recompile and run. What
happens? What if you reorder the send and receive calls in one of the
processes?
