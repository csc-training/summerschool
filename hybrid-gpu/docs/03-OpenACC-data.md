---
title:  OpenACC data management
author: CSC Summerschool
date:   2019-07
lang:   en
---


# OpenACC data environment

- OpenACC supports devices which either share memory with or have a
  separate memory from the host
- Constructs and clauses for
    - defining the variables on the device
    - transferring data to/from the device
- All variables used inside the `parallel` or `kernels` region will be
  treated as *implicit* variables if they are not present in any data
  clauses, i.e., copying to and from to the device is automatically
  performed


# Motivation for optimizing data movement

- Identify parallelism
- Express parallelism
- Express data movement
- Optimise loop performance
    - When dealing with an accelerator GPU device attached to a PCIe bus,
      **optimizing data movement** is often **essential** to achieving
      good performance

FIXME: missing figure


# Data lifetimes

- Typically data on the device has the same lifetime as the OpenACC
  construct (`parallel`, `kernels`, `data`) it is declared in
- It is possible to declare and refer to data residing statically on
  the device until deallocation takes place
- If the accelerator has a separate memory from the host, any
  modifications to the data on the host are not visible to the device
  before an explicit update has been made and vice versa


# Data constructs: data

- Define a region with data declared in the device memory
    - C/C++: `#pragma acc data [clauses]`
    - Fortran: `!$acc data [clauses]`
- Data transfers take place
    - from **the host** to **the device** upon entry to the region
    - from **the device** to **the host** upon exit from the region
- Functionality defined by *data clauses*
- *Data clauses* can also be used in `kernels` and `parallel` constructs


# Data construct: example

<div class="column">
## C/C++

```c
float a[100];
int iter;
int maxit=100;

#pragma acc data create(a)
{
    /* Initialize data on device */
    init(a);
    for (iter=0; iter < maxit; iter++) {
        /* Computations on device */
        acc_compute(a);
    }
}
```
</div>

<div class="column">
## Fortran

```fortran
real :: a(100)
integer :: iter
integer, parameter :: maxit = 100

!$acc data create(a)
! Initialize data on device
call init(a)
do iter=1,maxit
  ! Computations on device
  call acc_compute(a)
end do
!$acc end data
```
</div>


# Data constructs: data clauses

`present(var-list)`
  : **on entry/exit:** assume that memory is allocated and that data
    is present on the device
`create(var-list)`
  : **on entry:** allocate memory on the device, unless it was already
    present
  : **on exit:** deallocate memory on the device, if it was allocated
    on entry

- Technically **structured** reference count decremented, and deallocation
  happens if it goes to zero


# Data constructs: data clauses

`copy(var-list)`
  : **on entry:** If data is present on the device on entry, behave as
    with the `present` clause, otherwise allocate memory on the device
    and copy data from the host to the device.
  : **on exit:** copy data from the device to the host and deallocate
    memory on the device if it was allocated on entry
`copyin(var-list)`
  : **on entry:** same as `copy` on entry
  : **on exit:** deallocate memory on the device of it was allocated
    on entry
`copyout(var-list)`
  : **on entry:** If data is present on the device on entry, behave as
    with the `present` clause, otherwise allocate memory on the device
  : **on exit:** same as `copy` on exit


# Data constructs: data clauses

`reduction(operator:var-list)`
  : Performs reduction on the (scalar) variables in list
  : Private reduction variable is created for each gangs partial result
  : Private reduction variable is initialized to operators initial value
  : After parallel region the reduction operation is applied to private
    variables and result is aggregated to the shared variable *and* the
    aggregated result combined with the original value of the variable


# Reduction operators

FIXME: missing figure


# Data specification

- Data clauses specify functionality for different variables
- Overlapping data specifications are not allowed
- For array data, *array ranges* can be specified
  - C/C++: `arr[start_index:length]`, for instance `vec[0:n]`
  - Fortran: `arr(start_index:end_index)`, for instance `vec(1:n)`
  - Note: array data **must** be *contiguous* in memory (vectors,
    multidimensional arrays etc.)


# Default data environment in compute constructs

- All variables used inside the `parallel` or `kernels` region will be
  treated as *implicit* variables if they are not present in any data
  clauses, i.e., copying to and from to the device is automatically
  performed
- Implicit *array* variables are treated as having the `copy` clause in
  both cases
- Implicit *scalar* variables are treated as having the
    - `copy` clause in `kernels`
    - `firstprivate` clause in `parallel`


# `data` construct: example

<div class="column">
## C/C++

```c
int a[100], d[3][3], i, j;

#pragma acc data copy(a[0:100])
{
    #pragma acc parallel loop present(a)
    for (int i=0; i<100; i++)
        a[i] = a[i] + 1;
    #pragma acc parallel loop collapse(2) copyout(d)
    for (int i=0; i<3; ++i)
        for (int j=0; j<3; ++j)
            d[i][j] = i*3 + j + 1;
}
```
</div>

<div class="column">
## Fortran
```fortran
integer a(0:99), d(3,3), i, j

!$acc data copy(a(0:99))
  !$acc parallel loop present(a)
  do i=0,99
     a(i) = a(i) + 1
  end do
  !$acc end parallel loop
  !$acc parallel loop collapse(2) copyout(d)
  do j=1,3
     do i=1,3
        d(i,j) = i*3 + j + 1
     end do
  end do
  !$acc end parallel loop
!$acc end data
```
</div>


# Unstructured data regions

- Unstructured data regions enable one to handle cases where allocation
  and freeing is done in a different scope
- Useful for, e.g., C++ classes, Fortran modules
- `enter data` defines the start of an unstructured data region
    - C/C++: `#pragma acc enter data [clauses]`
    - Fortran: `!$acc enter data [clauses]`
- `exit data` defines the end of an unstructured data region
    - C/C++: `#pragma acc exit data [clauses]`
    - Fortran: `!$acc exit data [clauses]`


# Unstructured data

```c
class Vector {
    Vector(int n) : len(n) {
        v = new double[len];
        #pragma acc enter data create(v[0:len])
    }
    ~Vector() {
        #pragma acc exit data delete(v[0:len])
        delete[] v;
    }
    double v;
    int len;
};
```


# Enter data clauses

`if(condition)`
  : Do nothing if condition is false
`create(var-list)`
  : Allocate memory on the device
`copyin(var-list)`
  : Allocate memory on the device and copy data from the host to the
    device


# Exit data clauses

`if(condition)`
  : Do nothing if condition is false
`delete(var-list)`
  : Deallocate memory on the device
`copyout(var-list)`
  : Deallocate memory on the device and copy data from the device to the
    host

- Technically **dynamic** reference count decremented, and deallocation
  happens if it goes to zero


# Data directive: update

- Define variables to be updated within a data region between host and
  device memory
    - C/C++: `#pragma acc update [clauses]`
    - Fortran: `!$acc update [clauses]`
- Data transfer direction controlled by `host(var-list)` or
  `device(var-list)` clauses
    - `self` (`host`) clause updates variables from device to host
    - `device` clause updates variables from host to device
- At least one data direction clause must be present


# Data directive: update

- `update` is a single line executable directive
- Useful for producing snapshots of the device variables on the host or
  for updating variables on the device
    - Pass variables to host for visualization
    - Communication with other devices on other computing nodes
- Often used in conjunction with
    - Asynchronous execution of OpenACC constructs
    - Unstructured data regions


# `update` directive: example

<div class="column">
## C/C++
```c
float a[100];
int iter;
int maxit=100;

#pragma acc data create(a) {
    /* Initialize data on device */
    init(a);
    for (iter=0; iter < maxit; iter++) {
        /* Computations on device */
        acc_compute(a);
        #pragma acc update self(a) if(iter % 10 == 0)
    }
}
```
</div>

<div class="column">
## Fortran
```fortran
real :: a(100)
integer :: iter
integer, parameter :: maxit = 100

!$acc data create(a)
    ! Initialize data on device
    call init(a)
    do iter=1,maxit
        ! Computations on device
        call acc_compute(a)
        !$acc update self(a) if(mod(iter,10)==0)
    end do
!$acc end data
```
</div>


# Data directive: declare

- Makes a variable resident in accelerator memory
- Added at the declaration of a variable
- Data life-time on device is the implicit life-time of the variable
    - C/C++: `#pragma acc declare [clauses]`
    - Fortran: `!$acc declare [clauses]`
- Supports usual data clauses, and additionally
    - `device_resident`
    - `link`


# Porting and managed memory

<div class="column">
- Porting a code with complicated data structures can be challenging
  because every field in type has to be copied explicitly
- Recent GPUs have *Unified Memory* and support for page faults
</div>

<div class="column">
```c
typedef struct points {
    double x, y;
    int n;
}

void init_point() {
    points p;

    #pragma acc data create(p)
    {
        p.size = n;
        p.x = (double)malloc(...
        p.y = (double)malloc(...
        #pragma acc update device(p)
        #pragma acc copyin (p.x[0:n]...
```
</div>


# Managed memory

- Managed memory copies can be enabled using `--ta=tesla,cc60,managed`
  option on PGI compilers
- For full benefits Pascal generation GPU is needed
- Performance depends on the memory access patterns
    - For some cases performance is comparable with explicitly tuned
      versions

# Summary

- Data directive
    - Structured data region
    - Clauses: `copy`, `present`, `copyin`, `copyout`, `create`
- Enter data & exit data
    - Unstructured data region
- Update directive
- Declare directive
