<!--
SPDX-FileCopyrightText: 2010 CSC - IT Center for Science Ltd. <www.csc.fi>

SPDX-License-Identifier: CC-BY-4.0
-->

# Using asynchronous memory copies for multiple streams

Previously, kernels were launched concurrently in separate HIP streams, but memory copies back to the host were still blocking.

This exercise extends the previous example by performing device-to-host memory transfers asynchronously in each stream.

Start by **copying your last exercise result to this folder**:

```
cp ../04-streams-asynckernel/main.cpp .
```

Expected output is still the same:

```
1.000000 1.000000 1.000000 ...
1.313534 1.313534 1.313534 ...
0.739085 0.739085 0.739085 ...
```

## Instructions

In this exercise, you will:

- Replace blocking `hipMemcpy()` D2H calls with `hipMemcpyAsync()`
- Associate each memory transfer with its corresponding stream
- Manually synchronize your streams with the host before accessing host memory
- Inspect kernel and memory transfer overlap using ROCm profiling tools

Make sure to to move your synchronize calls to now happen **after** the device-to-host memory transfers.

## HIP functions used

The following HIP functions are needed in this exercise:

* `hipMemcpyAsync()`
* `hipStreamSynchronize()`

## Profiling kernel concurrency on LUMI

After completing the exercise, inspect asynchronous memory copy behavior through
`rocprof` and Perfetto.

Run the program with ROCm profiling enabled (replace normal srun launch command with the following):

```bash
srun  rocprofv3 --runtime-trace --output-format pftrace -- ./<yourapp>
```

This generates a file with a suffix: `.pftrace`, under a directory `nidXXXX`.

The directory identifier, `nidXXXX`, is based on the compute node you ran your program in.

Copy the file to your local machine:

```bash
scp <your_username>@lumi.csc.fi:/scratch/project_462001452/<your_username>/hip-programming/streams/04-streams-asyncmemcopy/nidXXXX/<xyz_results.pftrace> .
```

Replace the `<your_username>`, `nidXXXX` and `<xyz_results.pftrace>` sections in the above.  
The `.` at the end means that the file will be copied to the current directory.

You can open the trace in either:
- `chrome://tracing` (in Chromium)
- https://ui.perfetto.dev

In the timeline view, you should observe that the three kernels execute at overlapping times on the GPU and memory copies to host
can happen in each stream while computation in other kernels is ongoing.

For this small example, performance differences may be minimal, but the execution timeline behavior becomes visible in the profiler.

# Extra: Using pinned host memory

<details>
<summary><strong>Optional: Using pinned host memory with hipHostMalloc()</strong></summary>

By default, this exercise uses ordinary pageable host memory allocated with:

```cpp
malloc()
```

To achieve true asynchronous memory transfers and overlap between transfers and computation,
pinned (page-locked) host memory is typically required.

Try replacing your host memory allocations (for `a`, `b` and `c`):

```cpp
a = (float*) malloc(N_bytes);
```

with: 

```cpp
HIP_ERRCHK(hipHostMalloc((void**)&a, N_bytes));
```

and also:

```cpp
free(a);
```

with:

```cpp
HIP_ERRCHK(hipHostFree(a));
```

After making these changes, compile and run your program again, and inspect its trace.

</details>