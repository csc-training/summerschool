<!-- Adapted from material by EPCC https://github.com/EPCCed/archer2-MPI-2020-05-14 -->

# Broadcast and scatter

This exercises illustrates the collective communication operations and their benefits over point-to-point communication routines.
Your task is to write a program using send and receive calls to implement *broadcast* and *scatter* operations.

1. Examine, compile, and run the provided skeleton code with 4 MPI tasks ([skeleton.cpp](skeleton.cpp) or [skeleton.F90](skeleton.F90)).
   This code does the following:
   - Creates an integer array of size N (e.g. N=12) and initializes it using `init_buffer()` as follows:
      - rank 0: i (i=1,12)
      - other ranks: -1
   - Prints out the values in order in all the processes using `print_buffer()`
   You don't need to edit these helper functions.

2. Implement *broadcast*: rank 0 sends the contents of the entire array to other ranks.
   Print out the final values in all the processes to check the correctness of the results.
   1. Implement using the point-to-point `MPI_Send` and `MPI_Recv` operations.
   2. Implement using the collective `MPI_Bcast` operation.

3. Implement *scatter*: imagine that the array is divided into blocks of size N / P (P=number of processes),
   and rank 0 sends a different block to other ranks, so that the rank *i* receives the *i*th block.
   Print out the final values in all the processes to check the correctness of the results.
   1. Implement using the point-to-point `MPI_Send` and `MPI_Recv` operations.
   2. Implement using the collective `MPI_Scatter` operation.

4. Compare the timing between your point-to-point and collective implementations. How much faster the real collective operation is?
   Try increasing the array size to, e.g., N=10000, and running with a larger number of MPI tasks, e.g., 64 tasks.

5. (Bonus) How could you improve the timing of your send-recv implementation without still using the real collective call?
   See the [provided code example](solution/bcast-send-recv-tree.cpp) implementing broadcast using a tree algorithm.

