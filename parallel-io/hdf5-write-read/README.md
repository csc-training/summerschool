## HDF5 exercise

Complete the HDF5 exercises ([hdf5.c](hdf5.c) or [hdf5.f90](hdf5.f90))
by writing HDF5 writer and reader functions/subroutines.

On Mahti, you will need to load the module `hdf5/1.10.7-mpi` before you 
can compile the code:

```
module load hdf5/1.10.7-mpi
```
Now you shoud be able to compile with
```
mpicc hdf5.c -lhdf5
```
or
```
mpif90 hdf5.f90 -I/appl/spack/v017/install-tree/gcc-11.2.0/hdf5-1.10.7-qc3apk/include -lhdf5_fortran
```

After compiling and running the program. You can use the `h5dump` and `h5ls` 
commands to check the values in a HDF5 file.
