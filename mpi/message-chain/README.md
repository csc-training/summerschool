## Message chain

Write a program where every MPI task sends data to the next one.
Let `ntasks` be the number of the tasks, and `rank` the rank of the
current task. Your program should work as follows:

![](img/chain.svg)

- Every task with a rank less than `ntasks-1` sends a message to task
  `rank+1`. For example, task 0 sends a message to task 1.
- The message content is an integer array where each element is initialised to
  `rank`.
- The message tag is the receiver's rank.
- The sender prints out the number of elements it sends and the tag it used.
- All tasks with rank > 0 receive messages.
- Each receiver prints out their `rank` and the first element in the
  received array.

### Message chain with send and recv

1. Implement the program described above using `MPI_Send` and `MPI_Recv`. Utilize
   `MPI_PROC_NULL` when treating the special cases of
   the first and the last task in the chain so that all the `MPI_Send`s and
   `MPI_Recv`s are outside `if` statements. You
   may start from scratch or use the skeleton code
   ([chain.cpp](chain.cpp) or [chain.F90](chain.F90))
   as a starting point.

2. The skeleton code prints out the time spent in communication.
   Investigate the timings with different numbers of MPI tasks
   (e.g. 2, 4, 8, 16, ...), and pay attention especially to rank 0.
   Can you explain the behaviour?

3. Bonus: Use the status parameter to find out how much data was received,
   and print out this piece of information for all receivers.

### Message chain with sendrecv

4. Modify your program to use combined `MPI_Sendrecv` instead of individual
   `MPI_Send`s or `MPI_Recv`s. Keep a copy of the other version for comparison purposes.

   Investigate again the timings with different numbers of MPI tasks
   (e.g. 2, 4, 8, 16, ...). Compare the results to the implementation with individual
   `MPI_Send`s or `MPI_Recv`s and pay attention especially to rank 0.
   You should see significantly lower wait times for rank 0 in the `MPI_Sendrecv` version.
   Can you explain why?
