<!--
SPDX-FileCopyrightText: 2010 CSC - IT Center for Science Ltd. <www.csc.fi>

SPDX-License-Identifier: CC-BY-4.0
-->

# Executing multiple GPU kernels concurrently with HIP streams

This exercise demonstrates how multiple HIP streams can be used to execute independent GPU kernels concurrently. 

In the current code, all kernels are launched into the default stream.
This causes the kernels to execute sequentially.

If your program executes correctly, you should get the following output:

```
1.000000 1.000000 1.000000 ...
1.313534 1.313534 1.313534 ...
0.739085 0.739085 0.739085 ...
```

Printing the 10 first values in each kernel output array.

After running the code, you will use `rocprof` and Perfetto, to validate that your kernels are executing concurrently on the GPU.

Note that we do not execute any host to device memory copies in this
program, and all kernel data is generated directly on the GPU.

## Instructions

In this exercise, you will modify the `main()` function to:

- create three HIP streams
- launch one kernel into each stream
- synchronize the host with each stream
- destroy all streams

The GPU kernels are already implemented and do not need to be modified.

<details>
<summary><strong>Bonus exercise</strong></summary>

Start by having all kernels executing concurrently.

After this, modify the code so that `kernel_a` executes first fully and copies the memory for `d_a` back to host,
and have `kernel_b` and `kernel_c` execute asynchronously after this. Where should you put a synchronise call in the code?

</details>

## HIP functions used

The following HIP functions are needed in this exercise:

* `hipStreamCreate()`
* `hipStreamSynchronize()`
* `hipStreamDestroy()`

<details>
<summary><strong>Additional options</strong></summary>

In the exercise, you are instructed to use `hipStreamSynchronize()` to synchronize
your streams one by one. 

Another way to synchronize the GPU with the host is to wait for all GPU work at once using:

```
hipDeviceSynchronize()
```

Which synchronizes the entire device rather than a single stream.

</details>

## Profiling kernel concurrency on LUMI

After completing the exercise, validate that the kernels execute concurrently.

Launch your application with rocprofv3 at the end of your batch job script, by replacing your normal launch command with the following:

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
- `chrome://tracing` (in chromium)
- https://ui.perfetto.dev

## Understanding the kernels

<details>
<summary><strong>Understanding the GPU kernels in this exercise</strong></summary>

The kernels in this exercise are synthetic workloads only for teaching purposes.

Each GPU thread performs repeated floating-point computations using mathematical functions (e.g. `sin`, `cos`, `log`).

The workloads copy and operate on large arrays (~64 MB per array) and are quite heavy computationally (although quite redundant)
so that concurrent execution, as well as data transfers, become visible in the profiling tools.

</details>
