# HDF5 exercise instructions

In order to compile and run the HDF5 code examples and exercises you must have an MPI-aware HDF5 library available on your system, and instruct the compiler to link to `libhdf5`.

## HDF5 on CSC systems

- Lumi:
    1. `module load cray-hdf5/1.12.2.11`
    2. Compile `my_file.cpp` as `cc my_file.cpp -lhdf5`, or `ftn my_file.F90 -lhdf5_fortran` for Fortran code.
- Mahti:
    1. `module load hdf5/1.10.7-mpi`
    2. Compile with `mpicxx my_file.cpp -lhdf5`, or `mpif90 my_file.F90 -I/appl/spack/v017/install-tree/gcc-11.2.0/hdf5-1.10.7-qc3apk/include -lhdf5_fortran` for Fortran code.
You can run the resulting executable like any MPI program. The above modules also make the command line tools `h5dump` and `h5ls` available.


## For those who prefer testing and developing on their own systems

On many Linux distros you can obtain HDF5 development libraries through the system package manager. Eg. on Ubuntu you can use
```
apt install libhdf5-openmpi-dev hdf5-tools
```
`hdf5-tools` installs `h5dump`, `h5ls` which are needed for the exercises.

Note that `apt` also has package `libhdf5-dev` which provides `libhdf5-serial-dev`, ie. a serial non-MPI library. The parallel HDF5 exercises will not work with a serial implementation.

When compiling exercises, you will have to provide the compiler with `-lhdf5`, and possibly also include and linker paths if they aren't automatically set by the environment. Example on Ubuntu:
```bash
mpicxx hdf5-exercise.cpp -I/usr/include/hdf5/openmpi -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/ -lhdf5 -o hdf5-exercise.out
```
