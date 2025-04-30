# Investigating streams and events
**NOTE!** This code works only when compile with `hipcc --offload-arch=gfx90a streams.cpp`!
This exercise demonstrates an asynchronous data transfer and computation. Three different asynchronous cases are created, and their timings are printed out. The timings are recorded with hipEvent calls.

## Instructions

In the exercise, the following HIP functions are needed:

* `hipStreamCreate()`
* `hipMemcpyAsync()`
* `hipEventRecord()`
* `hipEventSynchronize()`
* `hipEventElapsedTime()`
* `hipStreamDestroy()`

### Case 0

1) Create and destroy `n_stream` streams in the main function in the locations marked by `#error`
2) The function `case_0()` is already complete and can be used as a reference

### Case 1

1) In the `case_1()` function, create a loop over `n_stream` and split the work done by the kernel call of Case 0 into multiple kernels calls (one kernel call per stream with an even workload per stream)
3) Record events using `start_event` and `stop_event` arrays for each stream before and after the kernel call

### Case 2

1) Create a loop into the function `case_2()`
	1) In the loop: Split the data copy from host to device into `n_stream` asynchronous memcopies. one for each stream (make sure the memcopies are split evenly for each stream)
	2) In the loop: Launch the kernel for each stream similarly to Case 1
	3) In the loop: Split the data copy from device to host into `n_stream` asynchronous memcopies. one for each stream (make sure the memcopies are split asynchronously 
2) Record total timing of the loop, use `start_event[n_stream]` and `stop_event[n_stream]` array positions
3) Additionally, record events for each stream using `start_event` and `stop_event` arrays before H-to-D memcopy and after D-to-H memcopy, respectively
4) Synchronize host with each `stop_event[i] `
5) Get timings between each corresponding `start_event[i]` and `stop_event[i]`

### Case 3

1) Copy the case 2 here
2) Instead of doing the asynchronous memcopies and the kernel in the same loop as in Case 2, create a separate loop for each (3 loops in total)
3) Make sure you record events in appropriate locations to get correct timings

## Additional considerations

* You can try setting `USE_PINNED_HOST_MEM` to `0` at line `#6`, to see how the timings change if we do not use pinned host memory.
