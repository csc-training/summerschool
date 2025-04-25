## Message exchange

In this exercise we practice basic message (data) passing between MPI processes.
Your task is to write a program where two processes send and receive a message to/from
each other using `MPI_Send` and `MPI_Recv`. You may assume that the program
is always ran as two MPI processes (`ntasks` is 2) and implement the message
passing between ranks 0 and 1.

The message content should be an integer array, where each element is
initialised to the rank of the process. After receiving a message, each
process should print out its own rank, number of elements it received,
and the first element in the received array.

Hints:

1. The `count` argument to `MPI_Recv` is the maximum number of elements to receive, NOT the actual message size.
Start by writing a simpler version that does not check the received message size at all.

2. If you run into deadlocks, remember that `MPI_Send` and `MPI_Recv` are blocking routines.
What should the order of sends and receives be to avoid deadlocks?

You may start from scratch or use one of the following as a starting point:
    - C: [exchange.c](exchange.c)
    - C++: [exchange.cpp](exchange.cpp)
    - Fortran: [exchange.F90](exchange.F90)

Try increasing the message size (e.g. to 100000), recompile and run. What
happens?

### Extra comment

Although `MPI_Send` and `MPI_Recv` are blocking, some MPI implementations
can allow the program to continue past `MPI_Send`.
even before the receive occurs (and vice versa). However, this is not guaranteed
and may depend on factors such as the message size.

Therefore you should always program so that no deadlocks can occur.

You can test deadlocking by changing the order of sends and receives in your solution.
It is best to use a large message size when testing this.
