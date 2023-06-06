## Message chain with non-blocking communication

Write a program where every MPI task sends data to the next one.
Let `ntasks` be the number of the tasks, and `myid` the rank of the
current task. Your program should work as follows:

- Every task with a rank less than `ntasks-1` sends a message to task
  `myid+1`. For example, task 0 sends a message to task 1.
- The message content is an integer array where each element is initialised to
  `myid`.
- The message tag is the receiver's rank.
- The sender prints out the number of elements it sends and the tag it used.
- All tasks with rank > 0 receive messages.
- Each receiver prints out their `myid` and the first element in the
  received array.

1. Implement the program described above using non-blocking communication, *i.e.* 
   `MPI_Isend`, `MPI_Irecv`, and `MPI_Wait`. Utilize
   `MPI_PROC_NULL` when treating the special cases of
   the first and the last task.
    You may start from scratch or use the skeleton code
   ([cpp/skeleton.cpp](cpp/skeleton.cpp) or
   [fortran/skeleton.F90](fortran/skeleton.F90))
   as a starting point. 

2. The skeleton code prints out the time spent in communication. 
   Investigate the timings with different numbers of MPI tasks 
   (e.g. 2, 4, 8, 16, ...). Compare the results to the implementation with
   `MPI_Send`s and `MPI_Recv`'s and pay attention 
   especially to rank 0. Can you explain the behaviour?

3. Write a version that uses `MPI_Waitall` instead of `MPI_Wait`s.
