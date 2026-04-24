<!--
SPDX-FileCopyrightText: 2026 CSC - IT Center for Science Ltd. <www.csc.fi>

SPDX-License-Identifier: CC-BY-4.0
-->

# Exercise: Reductions

In this exercise we will try to perform reductions with hip.

In the code we can find a naive approach that accumulate partial results in every thread and uses atomicAdd to sum them in memory. can we do better?
We also provide an example with the hipcub library (optimized for the task). How close can we go with "simple" optimization to that? We suggest a couple of techniques, in the [solution directory](solutions/) provide the code for those two kernels. you are however free to seek further optimizations with other techniques.


## Task: write a blockatomic kernel

1. in this first optimization, try to reduce the amount of atomicadd to global memory by performing the accumulation on the thread 0. atomicadds in shared memory are way faster, so the idea of this algorithm is that each thread does accumulation in his register, then does an atomicadd in shared memory. After all threads have done so, thread 0 will add this partial sum to the global memory.


## Task: use shared memory to further improve it

1. But we can do better with shared memory! what if we use a "tree" algorithm to reduce the partial sum from all threads to thread 0, instead of using atomicadds?

## [EXTRA] Task:
Try changing the base parameter values. what happens with repetitions? and with increasing/decreasing blocks? help yourself with profiler. 
[IDEAS to discuss] What about nvidia devices? does it behave the same way? 
[if i have time] shfl on cuda. shfl on AMD is implemented via shmem, so there is no real gain. is it the same on nvidia?

