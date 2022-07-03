---
title:  OpenMP interoperability with libraries
event:  CSC Summer School in High-Performance Computing 2022
lang:   en
---

# Interoperability with libraries {.section}


# Interoperability with libraries

- Often it may be useful to integrate the accelerated OpenMP code with
  other accelerated libraries
- MPI: MPI libraries are GPU-aware
- HIP/CUDA: It is possible to mix OpenMP and HIP/CUDA
    - Use OpenMP for memory management
    - Introduce OpenMP in existing GPU code
    - Use HIP/CUDA for tightest kernels, otherwise OpenMP
- Numerical GPU libraries: CUBLAS, CUFFT, MAGMA, CULA...
- Thrust, etc.


# Device data interoperability

- OpenMP includes methods to access the device data pointers in the
  host side
- Device data pointers can be used to interoperate with libraries and
  other programming techniques available for accelerator devices
    - HIP/CUDA kernels and libraries
    - GPU-aware MPI libraries
- Some features are still under active development, many things may not
  yet work as they are supposed to!


# Data constructs: `use_device_ptr` and `use_device_addr`

`omp target data use_device_ptr(var-list)`
  : `-`{.ghost}

- Define a device pointer to be available on the host
- Within the construct, all the pointer variables in `var-list`
  contain the device address

`omp target data use_device_addr(var-list)`
  : `-`{.ghost}

- Within the construct, all the variables in `var-list`
  have the address of the corresponding object in the device
- Can be used with non-pointer variables


# use_device_ptr: example with cublas

```c
cublasInit();
double *x, *y;
//Allocate x and y, and initialise x
#pragma omp target data map(to:x[:n]), map(from:y[:n]))
{
    #pragma omp target data use_device_ptr(x, y) {
        cublasDaxpy(n, a, x, 1, y, 1);
    }
}
```


# Calling CUDA-kernel from OpenMP-program

- In this scenario we have a (main) program written in C/C++ (or Fortran)
  and this driver uses OpenMP directives
    - CUDA-kernels must be called with help of OpenMP `use_device_ptr`
- Interface function in CUDA-file must have `extern "C" void func(...)`
- The CUDA-codes are compiled with NVIDIA `nvcc` compiler e.g.:
    - `nvcc -c -O3 --restrict daxpy_cuda.cu`
- The OpenMP-codes are compiled with NVIDIA `nvc` or `nvc++` compiler e.g.:
    - `nvc -c -mp=gpu -O3 call_cuda_from_openmp.c`
- For linking, `-lcudart -L$CUDA_HOME/lib64` is needed


# Calling CUDA/HIP-kernel from C OpenMP-program

<small>
<div class="column">
```c
// call_cuda_from_openmp.c

extern void daxpy(int n, double a,
                  const double *x, double *y);

int main(int argc, char *argv[])
{
    int n = (argc > 1) ? atoi(argv[1]) : (1 << 27);
    const double a = 2.0;
    double *x = malloc(n * sizeof(*x));
    double *y = malloc(n * sizeof(*y));

    #pragma omp target data map(alloc:x[0:n], y[0:n])
    {
        // Initialize x & y
        ...
        // Call CUDA-kernel
        #pragma omp target data use_device_ptr(x, y)
        daxpy(n, a, x, y);
        ...
    } // #pragma omp target data
```
</div>

<div class="column">
```c
// daxpy_cuda.cu

__global__
void daxpy_kernel(int n, double a,
                  const double *x, double *y)
{ // The actual CUDA-kernel
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n) {
        y[tid] += a * x[tid];
        tid += stride;
    }
}
extern "C" void daxpy(int n, double a,
                      const double *x, double *y)
{ // This can be called from C/C++ or Fortran
    dim3 blockdim = dim3(256, 1, 1);
    dim3 griddim = dim3(65536, 1, 1);
    daxpy_kernel<<<griddim, blockdim>>>(n, a, x, y);
}
```
</div>
</small>



# Calling CUDA/HIP-kernel from  Fortran OpenMP-program
<small>
<div class="column">
```c
// call_cuda/hip_from_openmp.f90
MODULE CUDA_INTERFACES
    INTERFACE
      subroutine f_daxpy(n, a, x, y) bind(C,name=daxpy)
      use iso_c_binding
      integer(c_int), value :: n
      double(c_double), value :: a
      real :: x(*), y(*)
    END INTERFACE
END MODULE CUDA_INTERFACES

...

// in the main programs
 use iso_c_binding
 ...
 integer(c_int) :: n
 double(c_double) :: a
 ...

!$omp target data use_device_ptr(x, y)

  call f_daxpy(n,a,x,y)

```

</div>

<div class="column">
```c
// call_cuda/hip_from_openmp.f90
MODULE CUDA_INTERFACES
    INTERFACE
      subroutine f_daxpy(n, a, x, y) bind(C,name=daxpy)
      use iso_c_binding
      integer(c_int), value :: n
      double(c_double), value :: a
      type(c_ptr), value :: x, y
    END INTERFACE
END MODULE CUDA_INTERFACES

...

// in the main programs
 use iso_c_binding
 ...
 integer(c_int) :: n
 double(c_double) :: a
 ...

!$omp target data use_device_ptr(x, y)

  call f_daxpy(n,a,c_loc(x),c_loc(y))

```
</div>
</small>


# Summary

- OpenMP programs can work in conjuction with GPU libraries or with
  own computational kernels written with lower level languages
  (e.g. CUDA/HIP).
- The pointer / reference to the data in device can be obtained with
  the `use_device_ptr` / `use_device_addr` clauses.
