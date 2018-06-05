## Parallel I/O with Posix

a) Write data from all MPI tasks to a single file using the spokesman
strategy. Gather data to a single MPI task and write it to a file. The
data should be kept in the order of the MPI ranks.

b) Verify the above write by reading the file using the spokesman
strategy. Use different number of MPI tasks than in writing.

c) Implement the above write so that all the MPI tasks write in to
separate files.  Skeleton codes are found in
[spokesman.c](c/spokesman.c) and
[spokesman_reader.c](c/spokesman_reader.c), or in
[spokesman.F90](fortran/spokesman.F90) and
[spokesman_reader.F90](fortran/spokesman_reader.F90)