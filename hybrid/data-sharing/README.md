## Data sharing and parallel regions

Explore data sharing between OpenMP threads.

Starting from the skeleton code ([variables.c](variables.c) or
[variables.F90](variables.F90)), add an OpenMP parallel region around the
block where the variables `Var1` and `Var2` are printed and manipulated.

What results do you get when you define the variables as **shared**,
**private** or **firstprivate**? Explain why do you get different results.
