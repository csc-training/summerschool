## HDF5-writerank exercise

Explore the HDF5 codes ([hdf5-writerank.c](hdf5-writerank.c) or [hdf5-writerank.F90](hdf5-writerank.F90)) and try to compile them.

On Lumi, you will need to load the module `cray-hdf5-parallel/1.12.2.7` before you 
can compile the code:

```
module load cray-hdf5-parallel/1.12.2.7
```
Now you shoud be able to compile with
```
cc hdf5-writerank.c -lhdf5
```
or
```
ftn hdf5-writerank.F90 -lhdf5_fortran
```

On Mahti, you will need to load the module `hdf5/1.10.7-mpi` before you 
can compile the code:

```
module load hdf5/1.10.7-mpi
```
Now you shoud be able to compile with
```
mpicc hdf5-writerank.c -lhdf5
```
or
```
mpif90 hdf5-writerank.F90 -I/appl/spack/v017/install-tree/gcc-11.2.0/hdf5-1.10.7-qc3apk/include -lhdf5_fortran
```

After compiling, try running the program with some number of MPI ranks. After this, try using the `h5dump` and `h5ls`  commands to check the values in the HDF5 file that is produced by running the example. Do you understand what the program does and what are the values in the file (as shown by `h5dump`)?