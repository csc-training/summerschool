# Calculating Fibonacci numbers in parallel

Fibonacci numbers are a sequence of integers defined by the recurrence 
relation 
 F<sub>n</sub> = F<sub>n-1</sub> + F<sub>n-2</sub>
with the initial values F<sub>0</sub>=0, F<sub>1</sub>=1.

The code [fibonacci.c](fibonacci.c) or [fibonacci.F90](fibonacci.F90)
contains (a very bad!!) recursive algorithm for calculating a
Fibonacci number n. Parallelize the code with OpenMP tasks.
