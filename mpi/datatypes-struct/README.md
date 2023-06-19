## Datatype for a struct / derived type

The skeleton code provided in
[struct.c](struct.c) / [struct.F90](struct.F90)
defines a struct (C) / derived type (Fortran).

1. Implement a custom MPI datatype, and send a single struct / derived type between two
   processes. Verify that the communication is performed succesfully.

2. Next, try to send an array of structs / derived types. Make sure that the *extent* of the
   datatype is correct (you may use `MPI_Type_get_extent` and `MPI_Get_address` for checking).

3. Implement the same send by sending just a stream of bytes (type `MPI_BYTE`).
   Verify correctness and compare the performance of these two approaches.
