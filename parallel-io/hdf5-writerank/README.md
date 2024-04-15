## HDF5-writerank exercise

Explore the HDF5 codes ([hdf5-writerank.c](hdf5-writerank.c) or [hdf5-writerank.f90](hdf5-writerank.f90)) and try to compile them.

On Lumi, you will need to load the module `cray-hdf5-parallel` before you 
can compile the code:

```
module load cray-hdf5-parallel
```

After compiling, try running the program with some number of MPI ranks. After this, try using the `h5dump` and `h5ls`  commands to check the values in the HDF5 file that is produced by running the example. Do you understand what the program does and what are the values in the file (as shown by `h5dump`)?