## Message chain

Write a simple program where every processor sends data to the next
one. Let ntasks be the number of the tasks, and myid the rank of the
current process. Your program should work as follows:

 - Every task with a rank less than ntasks-1 sends a message to task
myid+1.  For example, task 0 sends a message to task 1.

 - The message content is an integer array where each element is
   initialized to myid.

 - The message tag is the receiver’s id number.

 - The sender prints out the number of elements it sends and the tag
number.  All tasks with rank &ge; 1 receive messages.

- Each receiver prints out their myid, and the first element in the
  received array.

 a) Implement the program described above using `MPI_Send` and
 `MPI_Recv`. You may start from scratch or use as a starting point 
 [c/skeleton.c](c/skeleton.c) or [fortran/skeleton.F90](fortran/skeleton.F90)

 b) Rewrite the program using `MPI_Sendrecv` instead of `MPI_Send` and
`MPI_Recv`.

 c) (Bonus) Employ the `MPI_PROC_NULL` wildcard in treating the
 special cases of the first and last task in the chain.

 d) (Bonus) Use the status parameter to find out how much data was
 received, and print out this piece of information for all receivers.
