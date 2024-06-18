---
title:  OpenMP interoperability with libraries and HIP
event:  CSC Summer School in High-Performance Computing 2024
lang:   en
---

# OpenMP interoperability with libraries and HIP{.section}


# Interoperability with libraries

- Often it may be useful to integrate the accelerated OpenMP code with
  other accelerated libraries
- MPI: MPI libraries are GPU-aware
- HIP: It is possible to mix OpenMP and HIP
    - Use OpenMP for memory management
    - Introduce OpenMP in existing GPU code
    - Use HIP for tightest kernels, otherwise OpenMP
- Numerical GPU libraries: HIPBLAS, HIPFFT, ...
- Thrust, etc.


# Device data interoperability

- OpenMP includes methods to access the device data pointers in the
  host side
- Device data pointers can be used to interoperate with libraries and
  other programming techniques available for accelerator devices
    - HIP kernels and libraries
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


# use_device_ptr: example with hipblas

```c
// hipblas initialization
double *x, *y;
//Allocate x and y, and initialise x
#pragma omp target data map(to:x[:n]), map(from:y[:n]))
{
    #pragma omp target data use_device_ptr(x, y) {
        hipblasDaxpy(hipblashandler, a, x, 1, y, 1);
    }
}
```


# Calling HIP-kernel from OpenMP-program

- In this scenario we have a (main) program written in C/C++ (or Fortran)
  and this driver uses OpenMP directives
    - HIP-kernels must be called with help of OpenMP `use_device_ptr`
- Interface function in HIP-file must have `extern "C" void func(...)`
- The HIP-codes are compiled separetely with `hipcc` compiler e.g.:
    - `hipcc -c -O3 daxpy_hip.cu`
- The OpenMP-codes are compiled separetely  e.g.:
    - `ftn -c -fopenmp ... -O3 call_hip_from_openmp.f90`
- For linking, `-lamdhip64` is needed 


# Calling HIP-kernel from C OpenMP-program

<small>
<div class="column">
```c
// call_hip_from_openmp.c

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
        // Call HIP-kernel
        #pragma omp target data use_device_ptr(x, y)
        daxpy(n, a, x, y);
        ...
    } // #pragma omp target data
```
</div>

<div class="column">
```c
// daxpy_hip.cu

__global__
void daxpy_kernel(int n, double a,
                  const double *x, double *y)
{ // The actual HIP-kernel
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



# Calling HIP-kernel from  Fortran OpenMP-program
<small>
<div class="column">
```c
! call_hip_from_openmp.f90
MODULE HIP_INTERFACES
    INTERFACE
      subroutine f_daxpy(n, a, x, y) bind(C,name=daxpy)
      use iso_c_binding
      integer(c_int), value :: n
      double(c_double), value :: a
      type(c_ptr), value :: x, y
    END INTERFACE
END MODULE HIP_INTERFACES

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

<div class="column">

```c
// daxpy_hip.cu

__global__
void daxpy_kernel(int n, double a,
                  const double *x, double *y)
{ // The actual HIP-kernel
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


# Summary

- OpenMP programs can work in conjuction with GPU libraries or with
  own computational kernels written with lower level languages
  (e.g. HIP).
- The pointer / reference to the data in device can be obtained with
  the `use_device_ptr` / `use_device_addr` clauses.
