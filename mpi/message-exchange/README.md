## Simple message exchange

a) Write a simple program where two processes send and receive a
message to/from each other using `MPI_Send` and `MPI_Recv`. The message
content is an integer array, where each element is initialized to the
rank of the process. After receiving a message, each process should
print out the rank of the process and the first element in the
received array. You may start from scratch or use as a starting point
the skeleton code found in [message-exchange.c](message-exchange.c) or 
[message-exchange.F90](message-exchange.F90)

b) Increase the message size to 100,000, recompile and run.