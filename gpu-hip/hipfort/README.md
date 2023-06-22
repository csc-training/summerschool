# Hipfort: saxpy

Test hipfort by compiling and running a simple Fortran code that uses a HIP kernel to calculate saxpy on the GPU.

The following models are required:
```bash

module load LUMI/22.08
module load partition/G
module load cce/15.0.0
module load rocm/5.3.3
module use /project/project_465000536/EasyBuild/modules/LUMI/22.08/partition/G/
module load hipfort
```

Because the default `HIPFORT` installation only supports gfortran,  we us a custom module (installed via `EasyBuild`)  prepared in the summer school project. This package provide Fortran modules compatible with the Cray Fortran compiler as well as direct use of hipfort with the Fortran Cray Compiler wrapper (ftn).

Depending on the programmer needs, there are two way to compile the code. The first option is to use the AMD  provided `hipfc` compiler script:
```bash
hipfc -o main --offload-arch=gfx90a <hip_kernels>.cpp <fortran_code>.f90
```
The second option is use the Cray 'ftn' compiler wrapper as you would do to compile any fortran code. The appropriate module and library search paths as well as library linking flags will be automatically added by the compiler wrappper:
```bash
ftn -c <fortran_code>.f90
CC -xhip -c <hip_kernels>.cpp
ftn -o main <fortran_code>.o hip_kernels.o
```
The second option gives more flexibility when using a mix of OpenMP/OpenACC offloading to GPUs and HIP kernels/libraries.

## Equivalent CUDA Fortran code

For reference, file [cuda-fortran/main.cuf](cuda-fortran/main.cuf) contains an equivalent CUDA Fortran code.
