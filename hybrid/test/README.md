## Hello World ##

In this exercise you will test that you are able to compile and run
OpenMP application. Take a look at a simple example program in
[hello.c](hello.c) ([hello.F90](hello.F90) for
Fortran) that has been parallelized using OpenMP. It will first print
out a hello message (in serial) after which each thread will print out
an "X" character (in parallel).

Compile this program so that you enable OpenMP, and run it with one,
two and four threads. Do you get the expected amount of "X"s?
