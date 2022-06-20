## Hello world

Test that you are able to compile and run a program that uses OpenMP for
threading.

First, take a look at either the [C version](hello.c) or the [Fortran
version](hello.F90) of a simple example program that has been parallelised
using OpenMP threading. The program will print out a hello message (in serial)
followed by each thread printing out an "X" character (in parallel).

Second, compile the program so that you enable OpenMP and run it with 1, 2,
and 4 threads. Do you get the expected amount of Xs?
