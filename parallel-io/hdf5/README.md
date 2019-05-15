## HDF5 example

Study and test the HDF5 examples ([hdf5.c](hdf5.c) or [hdf5.f90](hdf5.f90))
where the [Simple MPI-IO exercise](../mpi-io) has been re-written with HDF5
using collective parallel write.

On Sisu, you will need to load the module `cray-hdf5-parallel` before you can
compile the code:

```
module load cray-hdf5-parallel
```

Compile and run the program. You can use the `h5dump` command to check the
values in a HDF5 file.
