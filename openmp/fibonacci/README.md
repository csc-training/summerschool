# Calculating Fibonacci numbers in parallel

A common use of OpenMP tasks is to parallelize recursive algorithms. This
exercise practices tasking with Fibonacci recursion.

Fibonacci numbers are a sequence of integers defined by the recurrence
relation $F_n = F_{n-1} + F_{n-2}$ with the initial values $F_0 = 0, F_1 = 1$.

The code [fibonacci.cpp](fibonacci.cpp) or [fibonacci.F90](fibonacci.F90)
contains (a very bad!!) recursive algorithm for calculating the Fibonacci
number $F_n$ for a given integer $n$.

Parallelize the code with OpenMP tasks.

## Comments
- The given code will fail to give correct results for $n \geq 47$ because of
integer overflow. You can focus on testing with $n$ smaller than this.
- The code prints out the time taken for computing $F_n$. A properly
multithreaded implementation should be faster for $n \gtrsim 35$, but likely
slower for smaller $n$ because of overhead associated with thread
and task spawning.
