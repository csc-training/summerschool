## Message exchange

In this exercise we practice basic message (data) passing between MPI processes.
Your task is to write a program where two processes send and receive a message to/from
each other using `MPI_Send` and `MPI_Recv`. You may assume that the program
is always ran as two MPI processes (`ntasks` is 2) and implement the message
passing between ranks 0 and 1.

The message content should be an integer array, where each element is
initialised to the rank of the process. After receiving a message, each
process should print out its own rank and the first element in the received array.

You may start from scratch or use one of the following as a starting point:

- C: [exchange.c](exchange.c)
- C++: [exchange.cpp](exchange.cpp)
- Fortran: [exchange.F90](exchange.F90)

Try increasing the message size (e.g. to 100000), recompile and run. Ensure your program still works;
if not, read the comment about deadlocks below.

Hints:

1. Pay close attention to the order in which the sends and receives should happen.

2. `MPI_Recv` requires a `status` argument, which can be used to obtain 'metadata'
about the incoming message. This is not necessary in this exercise,
and you may ignore the status altogether by passing `MPI_STATUS_IGNORE`.
The `status` argument is investigated further in a later exercise ([Probe message](../probe-message/)).


### Extra comment about deadlocking

Although `MPI_Send` and `MPI_Recv` are blocking, some MPI implementations
can allow the program to continue past `MPI_Send` even before the receive occurs.
However, this is not guaranteed and may depend on factors such as the message size.

You can test deadlocking by changing the order of sends and receives in your solution.
It is best to use a large message size when testing this.

#### Bonus technical explanation

The deeper explanation is that MPI defines several "communication modes" and has a different
send function for each mode (`MPI_Bsend` for buffered send, `MPI_Ssend` for synchronous etc.).

The MPI specification **only** mandates that the "standard mode" `MPI_Send` is blocking
in the sense that once this function call returns, the send buffer is safe to be modified
without changing the outgoing message. This means that implementations (`OpenMPI`, `MPICH`, ...)
are allowed allowed to choose the communication mode
on a case-by-case basis in an attempt to obtain better performance, and may choose to
eg. copy a small message in a new buffer and send it via a nonblocking send.

As MPI users we should *never* rely on internal, implementation-specific details
when programming message passing logic for our programs. In case of blocking communications,
always program so that no deadlocks can occur, eg. in this exercise, always assume that `MPI_Send` is truly blocking.

You can read more about MPI communication modes in section 3.4 of the [MPI specification](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report.pdf).
