## Message chain with Cartesian communicator

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

1. Create a Cartesian topology for the chain. Utilize MPI_Cart_shift for finding
   the neighbouring ranks and implement the communication with MPI point-to-point routines 
   (either blocking or non-blocking). Use
   [cpp/skeleton.cpp](skeleton.cpp) or [skeleton.F90](skeleton.F90) 
   as a starting point. 

2. Make a version where the chain is periodic, i.e. task `ntasks-1` sends to task 0
   and every task receives.
