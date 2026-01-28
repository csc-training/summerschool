<!--
SPDX-FileCopyrightText: 2010 CSC - IT Center for Science Ltd. <www.csc.fi>

SPDX-License-Identifier: CC-BY-4.0
-->

# Discussion

Note for instructors: use `run*.sh` scripts to generate all the output files.

## Task: Improve the performance

1. See `axpy-malloc.cpp`.

   Compile the code:

       g++ -std=c++20 -O3 -march=native -fopenmp axpy-malloc.cpp -o axpy.x

   Run with `$t` threads distributed across `$c` first cores on the node:

       export OMP_PLACES=cores OMP_PROC_BIND=spread OMP_DISPLAY_AFFINITY=true OMP_NUM_THREADS=$t
       srun -p test --exclusive --nodes=1 --ntasks-per-node=1 --cpus-per-task=$c -t 0:10:00 ./axpy.x 102400000

   Corresponding outputs are in files `axpy-t$t-c$c.out`. Note also the thread affinities printed in the files.

   As described in [Mahti documentation](https://docs.csc.fi/computing/systems-mahti/),
   a Mahti node has 2 CPU sockets, both of which have 4 NUMA domains, each of which has 2 CCDs, both of which have 2 CCXs, both of which have 4 cores.
   That is, there is 128 cores in total. Each NUMA domain has two memory controllers.

   The single thread case runs in 76 ms.

   If we limit the execution to a single NUMA domain (`c016` files), we see that runtime doesn't improve from the 60 ms of 2 threads when increasing the number of threads.
   As the axpy operation is memory bound, the execution time is limited by the memory bandwidth, and 2 threads seems to be able to saturate the two memory channels in a NUMA domain.

   Let's correlate the runtimes with the memory bandwidth. A single array is 0.763 MiB (`102400000*8/1024**3`).
   There is 2 reads and 1 write in a single axpy operation, so the total memory traffic is 38 GiB/s (`0.763*3/60e-3`).
   The maximum theoretical bandwidth for two memory channels of the corresponding [AMD EPYC 7H12](https://en.wikichip.org/wiki/amd/epyc/7h12) processor is 48 GiB/s. [^1]
   The achieved bandwidth is roughly 80% of the theoretical maximum, which is about the typical sustained bandwidth achievable in practice.

   When we take more NUMA domains in play with 2 threads each (`t002-c016`, `t004-c032`, `t008-c064`, `t016-c128`), we get nearly ideal speed up as the available memory bandwidth increases for each extra thread pair.

   Note! For more reliable timing, it'd be better to increase the array size further and collect more accurate statistics by running axpy in a loop.

2. See `axpy-unique_ptr.cpp` for an example.


[^1]: The theoretical maximum bandwidth can be calculated as follows.
      DDR4-3200 memory has clock speed of 1600 MHz and performs 2 data transfers per clock.
      Each data transfer is 8 bytes per memory channel (bus width).
      For two memory channels, the maximum bandwidth is then 47.68 GiB/s (`1600e6*2*8*2/1024**3`).)
