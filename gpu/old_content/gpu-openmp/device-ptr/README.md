## Using HIP library from OpenMP code

The provided codes [cpp/pi.c](cpp/pi.c) / [fortran.pi.F90](fortran/lumi/pi.F90)
calculate approximate value for **pi** by generating random numbers within a
square and calculating how many of them falls within a unit circle.

The code uses combination of OpenMP offloading and HIP library functions:

  - Memory management is done with OpenMP offload constructs
  - Random numbers are generated with `hiprand` library
  - Calculation of **pi** is done with OpenMP offload constructs

Add appropriate OpenMP constructs according the TODOs in the code. You can
compile the code in the provided Makefile by typing `make`.

(Fortran code uses the hipfort package for calling `hiprand` library. See also this [example](../../gpu-hip/hipfort/hiprand_example))
