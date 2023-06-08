## Collective operations

In this exercise we test different routines for collective communication.

Write a program for four MPI tasks. Each task should have a data vector with
the values initialised to:

![](img/sendbuffer.png)

In addition, each task has a receive buffer for eight elements and the
values in the buffer are initialized to -1.

Implement communication that sends and receives values from the data
vectors to the receive buffers using a single collective routine in
each case, so that the receive buffers will have the following values.
You can start from scratch or use the skeleton code
[skeleton.cpp](skeleton.cpp) or [skeleton.F90](skeleton.F90).

### Case 1

![](img/bcast.png)

### Case 2

![](img/scatter.png)

### Case 3

![](img/gatherv.png)

### Case 4

![](img/alltoall.png)

