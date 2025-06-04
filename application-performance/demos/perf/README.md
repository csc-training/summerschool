[Great examples on using perf](https://www.brendangregg.com/perf.html).

## Clone and compile the application

```bash
git clone https://github.com/cschpc/heat-equation.git
cd heat-equation/2d/mpi-openmp

CCFLAGS=-fno-omit-frame-pointer make CXX=CC CC=cc
```

## Demo

Get some counters from hybrid MPI-OpenMP run within a single node

Run the program on LUMI:
```bash
OMP_NUM_THREADS=16\
    srun\
        -A project_462000007\
        -N 1\
        -n 4\
        -c 16\
        -t 00:10:00\
        -p standard\
    ./heat_hybrid 8000 8000 20000
```

In a different shell, check on a job running on the node (e.g. with name `heat_hybrid`):
```bash
srun\
    -n 1\
    --overlap\
    --pty\
    --jobid="$(squeue --me -o "%.16A %.16j" | grep heat_hybrid | awk '{print $1}')"\
    $SHELL
```

To run `perf stat` with a given process name (e.g. `heat_hybrid`):
```bash
perf stat\
    -d -d -d\
    -p $(ps -u $USER | grep heat_hybrid | awk '{printf (NR == 1 ? "" : ",") $1;}')\
    --timeout 500
```

Get floating point vector operations
```bash
perf stat\
    -p $(ps -u $USER | grep heat_hybrid | awk '{printf (NR == 1 ? "" : ",") $1;}')\
    -e fp_ret_sse_avx_ops.all\
    -e fp_ret_sse_avx_ops.add_sub_flops\
    -e fp_ret_sse_avx_ops.div_flops\
    -e fp_ret_sse_avx_ops.mac_flops\
    -e fp_ret_sse_avx_ops.mult_flops\
    --timeout 500
```

## Demo

How to make a flamegraph of a program:
- compile with correct flags
- run with perf
- construct flamegraphs
- scp to laptop

```bash
OMP_NUM_THREADS=4\
    srun\
        -A project_462000007\
        -n 1\
        -c 4\
        -t 00:10:00\
        -p standard\
    perf record -F 1997 -g -o perf.data\
    ./heat_hybrid 2000 2000 1000
```

```bash
perf report --stdio -n -i perf.data
```

TODO: git clone flamegraph and generate on node
