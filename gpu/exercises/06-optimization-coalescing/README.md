# Performance counters and coalesced memory access

## Background and rocprof

`rocprof` can collect performance metric counters (`pmc`) of gpu kernels:
```bash
>rocprofv3 -i metrics.txt -o metrics.csv -- ./out
```

The counters to be collected are listed in the `metrics.txt` file and they are
outputted the `metrics.csv_counter_collection.csv` file. For example, if the file `metrics.txt` is

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

Every `pmc` line will be run separately, and the collection of the results will be
outputted into different folders.

*Note*: `rocprof --list-derived` and `rocprof --list-basic` must be
executed on a node with GPU present because it queries the available counters
from the hardware itself. [AMD Documentation on MI200/MI300 gpu counters](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300-mi200-performance-counters.html).

An MI250x GCD has 8 MiB of L2 memory shared across the CUs and each CU has 16
kiB of L1 memory.

In this exercise we will use the following metrics:

SQ_INSTS_VMEM_RD  Number of vector memory read instructions (including flat) issued. 
TCP_TCC_READ_REQ_sum  Total number of read requests to L2 cache
TCC_EA_RDREQ_sum     Number of TCC/EA read requests (either 32-byte or 64-byte). Sum over TCC instances.
TCC_EA_RDREQ_32B_sum Number of 32-byte TCC/EA read requests. Sum over TCC instances.
TCC_HIT_sum Total number of L2 cache hits.
TCC_MISS_sum Total number of L2 cache misses.
TCP_TOTAL_READ_sum Total number of vector L1d read accesses
TCP_TCC_READ_REQ_sum  Total number of read requests to L2 cach

equivalent and useful for write: 

TCC_EA_WRREQ_64B_sum  Number of 64-byte transactions going (64-byte write or CMPSWAP) over the TC_EA_wrreq interface 
TCC_EA_WRREQ_sum  Number of transactions (either 32-byte or 64-byte) going over the TC_EA_wrreq interface
SQ_INSTS_VMEM_WR  Number of vector memory write instructions (including flat) issued
TCP_TCC_WRITE_REQ_sum Total number of write requests to L2 cache


some terminology explanation:
TC -> texture cache (i.e. compute unit cache)
TCC -> L2 texture cache
TCP -> L1 texture cache
EX -> external agent (i.e. global memory)

## Exercise

The Code `copy.cpp` will read and write memory array of `4096*4096` float32
entries and various strides (`(1<<n)-1, n=1...21`) (`copy_kernel`).

In this exercise you will inspect how the impact of coalescing memory accesses affects the
performances of a kernel by looking at memory requests using the `rocprof` profiler.
The read request are "strided", while the writes are kept coalesced in this exercise.

- What do you think will be the best kernel? How do you expect the execution time to change when 
the stride increases, and why?
- How many L2 read requests are issued?
- How many device global memory read requests are issued?
- The number of Global memory read requests drop when the stride is around 4096. Why?
    *Hint*: Print out the values of `index` for some block in a 16x16 matrix.
    Are some of those indices adjacent within a warp?
- *Hint*: load and open the `metrics.csv` file with libreoffice or some other
  spreadsheet editor for quick manual analyses



