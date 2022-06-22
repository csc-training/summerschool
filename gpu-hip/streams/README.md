# Investigating streams and events

This exercise demonstrates an asynchronous data transfer and computation. Three different asynchronous cases are created, and their timings are printed out. The timings are recorded with hipEvent calls on the stream 0.

## Instructions

In the exercise, the following HIP functions are needed:

* hipStreamCreate()
* hipMemcpyAsync()
* hipLaunchKernelGGL()
* hipEventRecord()
* hipEventSynchronize()
* hipEventElapsedTime()
* hipStreamDestroy()

### Case 1

1) Create `nStream` (nStream = 4) streams and split the total problem size `n` evenly between these streams
2) Create a loop and execute the kernels in different streams with the appropriate workload per stream

### Case 2

1) Record a new event (use the existing variables where possible)
2) Use the existing streams, and split the problem size `n` evenly between these streams (same as case 1)
3) Create one loop 
	1) In the loop: Copy the data from host to device through the streams asynchronously (make sure the memcopies are split evenly for each stream)
	2) In the loop: Launch the kernel for each stream
	3) In the loop: Copy the data from device to host through the streams asynchronously 
4) Synchronize with hipEventSynchronize()
5) Record a stop event and calculate the new Elapsed time

### Case 3

1) Copy the case 2 here
2) Instead of doing the asynchronous memcopies and the kernel in the same loop, create a separate loop for each (3 in total)

## Additional considerations

* You can try setting `USE_PINNED_HOST_MEM` to `0` at line `#4`, to see how the timings change if we do not use pinned host memory.
