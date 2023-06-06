## Message chain with persistent communication

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

1. Implement the program described above using persistent communication, *i.e.*
   `MPI_Send_init`, `MPI_Recv_init`, `MPI_Start` and `MPI_Wait`.
   You may start from scratch or use the skeleton code
   ([skeleton.cpp](cpp/skeleton.cpp) or [skeleton.F90](fortran/skeleton.F90))
   as a starting point.

2. Write a version that uses `MPI_Startall` and `MPI_Waitall` instead of `MPI_Start`s and `MPI_Wait`s.
