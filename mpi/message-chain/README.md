## Message chain

Write a program where every MPI task sends data to the next one.
Let `ntasks` be the number of the tasks, and `myid` the rank of the
current task. Your program should work as follows:

![](img/chain.svg)

- Every task with a rank less than `ntasks-1` sends a message to task
  `myid+1`. For example, task 0 sends a message to task 1.
- The message content is an integer array where each element is initialised to
  `myid`.
- The message tag is the receiver's rank.
- The sender prints out the number of elements it sends and the tag it used.
- All tasks with rank > 0 receive messages.
- Each receiver prints out their `myid` and the first element in the
  received array.

1. Implement the program described above using `MPI_Send` and `MPI_Recv`. Utilize
   `MPI_PROC_NULL` when treating the special cases of
   the first and the last task in the chain so that all the `MPI_Send`s and
   `MPI_Recv`s are outside `if` statements. You
   may start from scratch or use the skeleton code
   ([cpp/skeleton.cpp](cpp/skeleton.cpp) or
   [fortran/skeleton.F90](fortran/skeleton.F90)) as a starting
   point.

2. The skeleton code prints out the time spent in communication.
   Investigate the timings with different numbers of MPI tasks
   (e.g. 2, 4, 8, 16, ...), and pay attention especially to rank 0.
   Can you explain the behaviour?

3. Use the status parameter to find out how much data was received,
   and print out this piece of information for all receivers.

4. Next, use combined `MPI_Sendrecv` instead of individual `MPI_Send`s or
   `MPI_Recv`s. Investigate again the timings with different numbers of MPI tasks
   (e.g. 2, 4, 8, 16, ...). Compare the results to the implementation with individual
   `MPI_Send`s or `MPI_Recv`s and pay attention especially to rank 0.
   Can you explain the behaviour?
