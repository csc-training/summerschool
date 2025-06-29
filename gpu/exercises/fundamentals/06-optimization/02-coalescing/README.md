# Performance counters and coalesced memory access

## Background and rocprof

`rocprof` can collect performance metric counters (`pmc`) of gpu kernels:
```bash
> rocprof -i metrics.txt -o metrics.csv ./copy
```

The counters to be collected are listed in the `metrics.txt` file and they are
outputted the `metrics.csv` file. For example, if the file `metrics.txt` is

```
pmc: VALUBusy, TCP_TCC_READ_REQ_sum
pmc: TCC_EA_RDREQ_sum
```
then `rocprof` will collect the derived metrics of how busy the vector
arithmetic logic units (VALU), how many L2 read requests are issued
(TCP_TCC_READ_REQ_sum) and how many global device memory read requests are
issued (TCC_EA_RDREQ_sum).

Here `TCP_TCC` refers to how many read requests the L1 (TCP) cache controller
issues to the L2 cache (TCC) and `TCC_EA` refers to how many reads L2 cache
controller issues to the interconnect (`EA`).

The options `--list-derived` and `--list-basic` will list the available derived
and basic counters. 

*Note*: `rocprof --list-derived` and `rocprof --list-basic` must be
executed on a node with GPU present because it queries the available counters
from the hardware itself. [AMD Documentation on MI200/MI300 gpu counters](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300-mi200-performance-counters.html).

An MI250x GCD has 8 MiB of L2 memory shared across the CUs and each CU has 16
kiB of L1 memory.

## Exercise

The Code `copy.cpp` will read and write memory array of 4096*4096 float32
entries and various strides (`(1<<n)-1, n=1...21`) (`copy_kernel` and line
'59`).

In this exercise you will inspect how well the GPU is able to coalesce the
memory requests using the `rocprof` profiler.

- How many L2 read requests (64 B and 32 B combined) are issued?
- How many device global memory read requests (64 B and 32 B combined) are
  issued?
- The number of L2 read requests drop when the stride is around 4096. Why?
    *Hint*: Print out the values of `index` for some block in a 16x16 matrix.
    Are some of those indices adjacent within a warp?
- *Hint*: load and open the `metrics.csv` file with libreoffice or some other
  spreadsheet editor for quick manual analyses
