# parallel-write

In this exercise we practice outputting data to disk in a parallel MPI program.
The goal is to implement file writing first using standard I/O routines combined with MPI communication,
and then again using a collective MPI-IO parallel write routine.

The exercise tries to mimic a typical I/O situation in HPC simulations where the data to be stored is distributed
across many MPI processes, and we want to write it to disk in some well-defined order.
For example, if the data represents values of some simulated quantity everywhere on a distributed grid,
we may want to write the data in the same order as the grid points are indexed.

Have a look at the following unfinished code:

- [`parallel-write.cpp`](./parallel-write.cpp) (C/C++)
- TODO fortran

Here each MPI process allocates a "local data" array of integers and we wish to write the full data to a **single** file on disk.
The resulting file should be ordered so that data from rank 0 comes first, then data from rank 1 *etc*.

## Part 1

Your task is to implement the two unfinished functions in the code:
1. `single_writer()` should perform the file I/O using the "spokesperson" strategy,
ie. all data is collected to MPI rank 0, which then writes it to file using standard library write routines.
2. `collective_write()` should perform a parallel write using the collective MPI-IO routine `MPI_File_write_at_all()`.

The starting code has two global constants that you can adjust for testing:
- `numElements` specifies how many integers will be written in total. Must be divisible by the number of MPI tasks.
- The `doDebugPrint` boolean can be used to enable or disable printing of file contents for debugging purposes.
Note that the debug prints are not very useful if `numElements` is very large. For this reason the starting code enables debug prints only if `numElements <= 100`.

The provided `main` function calls both of the write functions and also reads and dumps contents of the written files. Use this "debug" output to check that your writes are correct.

**Example file contents** (when `numElements` is 32):
- 4 MPI tasks:
```
00000000111111112222222233333333
```
- 8 MPI tasks:
```
00001111222233334444555566667777
```

## Part 2 (bonus)

Let's compare the I/O performance of the two implementations:
- Ensure `doDebugPrint` is set to `false` in order to keep the output simple.
- Add timings to the `main` function for measuring the evaluation time of `single_writer()` and `collective_write()` separately. You can use the `MPI_Wtime()` function for timestamps (in seconds). Print the evaluation time from MPI rank 0.
- Repeat the time measurements a few times (eg. 5) by putting them in a loop, printing the timings on each iteration.

Run and compare the timings. Which implementation is faster?
- For `collective_write()`, you should notice that its first invocation is considerably slower than the consecutive calls. This is because the MPI-IO library has to initialize internal state on its first use.

Then increase `numElements` to a larger value so that more data will be written. Recompile and run with varying number of MPI tasks.
Try eg. the following combinations:
- `numElements == 1024`, 2 and 4 MPI tasks
- `numElements == 1048576` ($1024^2$), 4 and 8 MPI tasks
- `numElements == 104857600` ($1024^2 \times 100$), 8 and 16 MPI tasks

You should find that the `collective_write()` method eventually becomes faster than `single_writer()` once the data size is large enough. Can you explain this behavior?
