## HDF5 exercise

Complete the HDF5 exercises ([hdf5.c](hdf5.c) or [hdf5.f90](hdf5.f90))
by writing HDF5 writer and reader functions/subroutines.

On Lumi, you will need to load the module `cray-hdf5-parallel` before you 
can compile the code:

```
module load cray-hdf5-parallel
```

After compiling and running the program. You can use the `h5dump` and `h5ls` 
commands to check the values in a HDF5 file.
