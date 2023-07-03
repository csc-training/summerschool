# Hipfort

## Usage 

Test hipfort by compiling and running a simple Fortran code that uses a HIP kernel to calculate saxpy on the GPU.

The following modules are required:
```bash

module load LUMI/22.08
module load partition/G
module load cce/15.0.1
module load rocm/5.3.3
```

Because the default `HIPFORT` installation only supports gfortran,  we use a custom installation  prepared in the summer school project. This package provide Fortran modules compatible with the Cray Fortran compiler as well as direct use of hipfort with the Fortran Cray Compiler wrapper (ftn). 

The package was installed via:
```bash
git clone https://github.com/ROCmSoftwarePlatform/hipfort.git
cd hipfort;
mkdir build;
cd build;
cmake -DHIPFORT_INSTALL_DIR=<path-to>/HIPFORT -DHIPFORT_COMPILER_FLAGS="-ffree -eZ" -DHIPFORT_COMPILER=ftn -DHIPFORT_AR=${CRAY_BINUTILS_BIN_X86_64}/ar -DHIPFORT_RANLIB=${CRAY_BINUTILS_BIN_X86_64}/ranlib  ..
make -j 64 
make install
```

We will use the Cray 'ftn' compiler wrapper as you would do to compile any fortran code plus some additional flags:
```bash
export HIPFORT_HOME=/project/project_465000536/appl/HIPFORT
ftn -I$HIPFORT_HOME/include/hipfort/amdgcn "-DHIPFORT_ARCH=\"amd\"" -L$HIPFORT_HOME/lib -lhipfort-amdgcn $LIB_FLAGS -c <fortran_code>.f90 
CC -xhip -c <hip_kernels>.cpp
ftn  -I$HIPFORT_HOME/include/hipfort/amdgcn "-DHIPFORT_ARCH=\"amd\"" -L$HIPFORT_HOME/lib -lhipfort-amdgcn $LIB_FLAGS -o main <fortran_code>.o hip_kernels.o
```
This option gives enough flexibility for calling HIP libraries from Fortran or for a mix of OpenMP/OpenACC offloading to GPUs and HIP kernels/libraries.

## Examples: `saxpy` and `pi` 

There are two examples for exploring. 

The first one [`saxpy`](saxpy/) demonstrates a basic memory management and  how to call from fortran a `hip` kernel. For reference, file [cuda-fortran/main.cuf](saxpy/cuda-fortran/main.cuf) contains an equivalent CUDA Fortran code. 


Hipfort provides interfaces for various highly optimized library. 

The folder  [hiprand_example](hiprand_example/) shows how to call the `hiprand` for generating single precision uniform random distributed nubmbers for calculation the value of `pi`.

The exercise is to analyse and run the programs. For more examples of hipfort check also the [official repository](https://github.com/ROCmSoftwarePlatform/hipfort/tree/develop/test).

## Heat Equation
Starting from the Fortran, serial or MPI, heat equation offload to GPU the update loops. 
