<!--
SPDX-FileCopyrightText: 2010 CSC - IT Center for Science Ltd. <www.csc.fi>

SPDX-License-Identifier: CC-BY-4.0
-->

# Overlapping CPU and GPU work

This exercise demonstrates the asynchronocity between the GPU and CPU, and how arbitrary CPU function execution can overlap with GPU kernel execution.

A GPU kernel launch is asynchronous with respect to the host. This means that after the CPU launches a GPU kernel, the CPU can continue executing other independent work.

In this exercise, you compare two versions of the same workflow:

- `dontOverlapCpuGpuWork()` launches a GPU kernel, synchronizes immediately, and only then performs CPU work
- `overlapCpuGpuWork()` launches a GPU kernel, performs independent CPU work while the GPU kernel is running, and synchronizes only when the GPU result is needed

Both versions compute the same results, but in the second version the goal is to overlap these computations, leading to a quicker total execution time.

If your program executes correctly, you should get output similar to:

```text
The results are OK! (XYZ.XYZ ms - Execute CPU/GPU work sequentially)
The results are OK! (ZYX.ZYX ms - Overlap CPU/GPU work)
```

## Instructions

In this exercise, you will:

- launch a GPU kernel asynchronously into a HIP stream
- synchronize immediately in one function before doing CPU work
- continue doing CPU work in the other function before synchronizing
- compare the timing between the two synchronization strategies
- inspect CPU/GPU overlap using `rocprof`, rocTX ranges, and Perfetto

Only the following HIP function is needed in this exercise:

- `hipStreamSynchronize()`

### Compiling

**Note!** This bonus exercise uses ROCm-specific profiling tools and ROCTx ranges. It is intended for LUMI during the course.

The profiling instructions will not work on Mahti as written, because Mahti uses NVIDIA GPUs. To run a similar profiling exercise on Mahti, the ROCTx annotations and `rocprofv3` commands would need to be replaced with NVIDIA tools, e.g. NVTX annotations and Nsys.

This is a bonus exercise that implements `roctX` ranges to make CPU work visible in the profiling timeline:

- `roctxRangePush()`
- `roctxRangePop()`

Compile the program with `rocTX` linked:

```bash
CC -xhip -lroctx64 main.cpp -o prog.x
```

## Profiling CPU/GPU overlap

Run the program with ROCm profiling **and** `roctx` enabled:

Launch your application with rocprofv3 **and** `roctx` enabled, by replacing your normal launch command at the end of your job script with the following:

```bash
srun rocprofv3 --runtime-trace --marker-trace --output-format pftrace -- ./<yourapp>
```

This generates a file with a suffix: `.pftrace`, under a directory `nidXXXX`.

Copy the file to your local machine:

```bash
scp <your_username>@lumi.csc.fi:/scratch/project_462001452/<your_username>/summerschool/gpu/exercises/04-bonus-cpu-gpu-overlap/<path-to-your-file>.pftrace .
```

Replace the `<your_username>` and `<path-to-your-file>` sections in the above.  
The `.` at the end means that the file will be copied to the current directory.

You can open the trace in either:

- `chrome://tracing` in Chromium
- https://ui.perfetto.dev

## Background

<details>
<summary><strong>CPU/GPU concurrent execution</strong></summary>

GPU kernel launches are asynchronous with respect to the host.

This means that when the CPU launches a kernel:

```cpp
gpu_kernel<<<grid, block, 0, stream>>>(d_A, n);
```

the launch usually returns before the GPU kernel has finished. The CPU can continue executing other work.

A simple but often inefficient approach is to synchronize immediately:

```cpp
gpu_kernel<<<grid, block, 0, stream>>>(d_A, n);

hipStreamSynchronize(stream);

cpu_work();
```

This is correct, but it forces the CPU to wait before doing its own work.

If the CPU work is independent of the GPU result, the synchronization can be delayed:

```cpp
gpu_kernel<<<grid, block, 0, stream>>>(d_A, n);

cpu_work();

hipStreamSynchronize(stream);
```

Now CPU work and GPU work can overlap.

The expected timing behavior is approximately:

```text
synchronize immediately  ~= GPU work + CPU work
overlap CPU/GPU work     ~= max(GPU work, CPU work)
```

In practice, exact timings depend on system load, runtime overheads, and how well the CPU and GPU workloads are balanced.

</details>

<details>
<summary><strong>rocTX ranges</strong></summary>

`rocprof --hip-trace` shows HIP API calls and GPU activity, such as kernel launches and kernel execution.

However, ordinary CPU code does not automatically appear as a visible region in the trace. For example, a function such as:

```cpp
compute_on_cpu(...);
```

may take time, but without extra annotations it appears in the trace as empty space between HIP API calls.

rocTX ranges are extra annotations that mark regions of host code:

```cpp
roctxRangePush("CPU executing");
compute_on_cpu(...);
roctxRangePop();
```

When the program is profiled with:

```bash
run_tue rocprofv3 --runtime-trace --marker-trace --output-format pftrace -- ./<yourapp>
```

these ranges appear in the trace. This makes it easier to see when the CPU is doing work and when that work overlaps with GPU execution.

In this exercise, rocTX is used only for visualization for the profile. It does not affect program correctness and only adds minimal performance overhead.

</details>

## Bonus: What do the CPU and GPU computations calculate?

<details><summary>GPU/CPU computations in the exercise</summary>

Both the CPU and GPU workloads repeatedly evaluate the trigonometric identity:

$$
\sin^2(x) + \cos^2(x) = 1
$$

The GPU writes values close to `1.0f` into the output array.

The CPU computes an average value that should also be close to `1.0`.

The function `checkResults()` verifies that both the GPU and CPU results are close to the expected value.


</details>
