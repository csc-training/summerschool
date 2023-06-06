## Collective communications in Heat equation solver

Utilize collective communication in heat equation code.

1. Replace the individual sends and receives in the routine `average` (in [fortran/utilities.F90](fortran/utilities.F90) or in [cpp/utilities.cpp](cpp/utilities.cpp)) with appropriate collective
communication.
2. Replace the individual sends and receives in the routine `read_field` (in [fortran/io.F90](fortran/io.F90) or in [cpp/io.cpp](cpp/io.cpp)) with appropriate collective communication. Note that the code needs to be run with the initial data read from an input file (found under the [common](common) directory), *i.e.*
  ```
  srun ./heat_mpi bottle_dat
  ```

3. Is it possible to use collective communications also in the routine `write_field` (in [fortran/io.F90](fortran/io.F90) or in [cpp/io.cpp](cpp/io.cpp))?


Note that either the [base exercise](README.md) or exercise with [MPI_Sendrecv](READ_sendrecv.md)
needs to be completed for this exercise.

