## HDF5 example

Study and test the HDF5 examples ([hdf5.c](hdf5.c) or [hdf5.f90](hdf5.f90))
where the [Simple MPI-IO exercise](../mpi-io) has been re-written with HDF5
using collective parallel write.

On Puhti, you will need to load the module `hdf5/1.10.7-mpi` before you can
compile the code:

```
module load hdf5/1.10.7-mpi
```

Compile and run the program. You can use the `h5dump` command to check the
values in a HDF5 file.
