# Calculating Fibonacci numbers in parallel

Fibonacci numbers are a sequence of integers defined by the recurrence
relation $F_n = F_{n-1} + F_{n-2}$ with the initial values $F_0 = 0, F_n = 1$.

The code [fibonacci.cpp](fibonacci.cpp) or [fibonacci.F90](fibonacci.F90)
contains (a very bad!!) recursive algorithm for calculating a
Fibonacci number $n$. Parallelize the code with OpenMP tasks.
