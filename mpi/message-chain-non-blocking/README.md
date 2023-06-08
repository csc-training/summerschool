## Message chain

See [the earlier message chain exercise](../message-chain) for the description of the message chain.

### Message chain with non-blocking communication

1. Implement the program using non-blocking communication, *i.e.*
   `MPI_Isend`, `MPI_Irecv`, and `MPI_Wait`. Utilize
   `MPI_PROC_NULL` when treating the special cases of
   the first and the last task.
   You may start from scratch, or use the skeleton code or
   your solution from [the earlier message chain exercise](../message-chain)
   as a starting point.

2. The skeleton code prints out the time spent in communication.
   Investigate the timings with different numbers of MPI tasks
   (e.g. 2, 4, 8, 16, ...). Compare the results to the implementation with
   `MPI_Send`s and `MPI_Recv`'s and pay attention
   especially to rank 0. Can you explain the behaviour?

3. Write a version that uses `MPI_Waitall` instead of `MPI_Wait`s.


### Message chain with persistent communication

1. Implement the program using persistent communication, *i.e.*
   `MPI_Send_init`, `MPI_Recv_init`, `MPI_Start` and `MPI_Wait`.
   You may start from scratch, or use the skeleton code or
   your solution from [the earlier message chain exercise](../message-chain)
   as a starting point.

2. Write a version that uses `MPI_Startall` and `MPI_Waitall` instead of `MPI_Start`s and `MPI_Wait`s.
