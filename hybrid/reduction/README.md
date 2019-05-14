## Reduction and critical ##

Continue with the previous example
[../race-condition/](../race-condition/) and use `reduction` clause to
compute the sum correctly.

Implement also an alternative version where each thread computes its
own part to a private variable and the use a `critical` section after
the loop to compute the global sum.

Try to compile and run your code also without OpenMP. An example
solution can be found in
[../race-condition/solution](../race-condition/solution).
