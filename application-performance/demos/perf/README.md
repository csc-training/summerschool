[Great examples on using perf](https://www.brendangregg.com/perf.html).

## Clone and compile the application

```bash
git clone https://github.com/cschpc/heat-equation.git
cd heat-equation/2d/mpi-openmp

#ml LUMI/23.02
#ml partition/C
#ml craype-huge2M

CCFLAGS=-fno-omit-frame-pointer make CXX=CC CC=cc
```

## Demo

Show how to get some simple counters from a running program:
- ssh to the node
- find the pid
- check how it's doing

To view the command before running it (e.g. to see there's an actual PID):
`ps -C <process name> | tail -n 1 | awk '{print "perf stat -d -p " $1 " -- sleep 5";}'`

To run it, pipe it to shell:
`ps -C <process name> | tail -n 1 | awk '{print "perf stat -d -p " $1 " -- sleep 5";}' | sh`

## Demo

How to make a flamegraph of a program:
- compile with correct flags
- run with perf
- construct flamegraphs
- scp to laptop

`perf record -F 999 -g mpirun -n 2 ./heat_hybrid`
`perf report --stdio -n -i perf.data`
