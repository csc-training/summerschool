# Histogram bin count with HIP

In this exercise we will see different versions of an histogram count program, starting from the serial implementation to advanced and more performant.

[exercise instructions](../../../../../exercise-instructions.md).

run and profile the basic program

how can we parallelize it?
first step. make every block "own" a bin, thread scan different cell of the array in parallel and add to local bin if value is corresponding. at the end of the scan we make a reduction in the block, then we write in the histogram

Pros: atomics‑free and parallel within each block.
Cons: Each block scans all N elements → very expensive for large num_bins.

how can we reduce the amount of scans? yes, but we need a way to ensure that there are no data races. Atomic_add comes to help!
second step. use atomic add.

third step. can we do better? atomic_add is an expensive operation when done to global memory...
