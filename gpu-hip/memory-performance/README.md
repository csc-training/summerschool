# Memory performance tests

The purpose of this exercise is to show the difference in performance of different ways to manage the memory.
The kernel does the same thing in two different ways, on the one hand it will use coalesced memory accesses while in another circumstance it won't.
The kernel will be called with 3 different strategies: no recurring allocations, recurring allocations and asynchronous recurring allocations

The task is to fill the missing function calls in the code indicated by lines beginning with `#error`, and followed by a descriptive instruction.

To compile this, the usage of c++17 is required, so the application must be compiled with `CC -xhip --std=c++17 performance.cpp`. Moreover, rocm/5.3.3 or higher is needed, older rocm modules do not support memory asynchronous malloc/free (they hang)
