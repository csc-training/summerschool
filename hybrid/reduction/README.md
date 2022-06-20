## Reduction and critical

Fix the model "solution" of the [previous exercise](../race-condition/) that
still contains a race condition ([sum.c](../race-condition/solution/sum.c) or
[sum.F90](../race-condition/solution/sum.F90)).

1. Use `reduction` clause to compute the sum correctly.

2. (Bonus) Implement an alternative version where each thread computes its
   own part to a private variable and the use a `critical` section after
   the loop to compute the global sum.

Try to compile and run your code also without OpenMP.
