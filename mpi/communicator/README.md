## Communicators and collectives

In this exercise we combine collective communcation with user defined
communicators. Write a program for four MPI processes, such that each
process has a data vector with the following data:

![](img/sendbuffer.png)

In addition, each task has a receive buffer for eight elements and the
values in the buffer are initialized to -1.

Implement now a pattern with user defined communicators and collective
operation so that the receive buffers will have the following values:

![](img/comm-split-reduce.svg)

You can start from scratch or use the skeleton code
[skeleton.cpp](skeleton.cpp) or [skeleton.F90](skeleton.F90).

**HINT:** Notice how the desired result in Task 0 can be achieved by adding the send arrays in tasks 0 and 1 element wise. What about Task 2?
