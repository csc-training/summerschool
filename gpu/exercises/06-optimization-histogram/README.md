<!--
SPDX-FileCopyrightText: 2026 CSC - IT Center for Science Ltd. <www.csc.fi>

SPDX-License-Identifier: CC-BY-4.0
-->

# Exercise: Histogram bin count with HIP
In this exercise we will see different versions of an histogram count program, starting from the serial implementation to advanced and more performant. For the general instructions, you can look at [exercise instructions](../../../exercise-instructions-lumi.md).

## Task: run and profile the basic program. 
In the provided code, the only provided kernel is a very naive one where a single thread is working. This is even slower than CPU, so we need to find some parallelism in order to leverage GPU capabilities.

how can we parallelize it?

## Task: first attempt: no AtomicAdd
Write a kernel where every block "owns" a bin, thread scan different cell of the array in parallel and add to local bin if value is corresponding. at the end of the scan we make a reduction in the block, then we write in the histogram

Pros: atomics‑free and parallel within each block.

Cons: Each block scans all N elements → very expensive for large num_bins.

## Task: Reduce number of scans
Can we reduce the amount of scans? yes, but we need a way to ensure that there are no data races. Atomic_add comes to help! With that we can scan all numbers once, and add them to the correct bin only. This greatly reduces the amount of needed read from the memory!

## Task: Can we do better?
atomic_add is an expensive operation when done to global memory... Can we reduce the amount of atomic add to global memory, and make most of them to shared memory instead?
