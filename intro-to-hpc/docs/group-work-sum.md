---
title:  "Group work: parallel sum"
event:  CSC Summer School in High-Performance Computing 2023
lang:   en
---

# Calculating sum in parallel

<small>

- Task: calculate sum of 20 numbers
- Assumptions: 
    - Summing up two numbers takes **1 s** and communicating single number takes **0.1 s** 
    - No time is needed for setting up the problem and distributing work
- Work out with pen and paper how much time is needed when the work is performed with 1, 2, 4, and 
  8 workers. What is the speed-up with different number of workers?
- Discuss what limits parallel scaling and how parallel efficiency could be improved.
- Does the situation change if the task is to calculate 1020 numbers? What is now 
  the parallel speed-up with eight workers?

</small>

<!-- "Model" solutions

Assume: no overlap of communication, "serial" reduction

1 worker:           19 s
2 workers:  9 + 1 (reduction) + 0.1 (comm) = 10.1 s  speedup = 1.9
4 workers:  4 + 3 (reduction) + 0.3 (comm) =  7.3 s  speedup = 1.4
With 8 workers there is load imbalance: some workers need to sum 3 numbers, some 2
8 workers:  2 + 7 (reduction) + 0.7 (comm) =  9.7 s

1020 numbers:
1 worker:           1019 s
8 workers:  128 + 7 (reduction) + 0.7 (comm) = 135.7  speedup = 7.5

-->

