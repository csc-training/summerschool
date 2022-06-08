---
title:  "OpenMP unstructured data regions"
author: CSC - IT Center for Science
date:   2022-06
lang:   en
---


# Unstructured data regions {.section}

# Unstructured data regions

- Unstructured data regions enable one to handle cases where allocation
  and freeing is done in a different scope
- Useful for e.g. C++ classes, Fortran modules
- `enter data` defines the start of an unstructured data region
    - C/C++: `#pragma omp enter data [clauses]`
    - Fortran: `!$omp enter data [clauses]`
- `exit data` defines the end of an unstructured data region
    - C/C++: `#pragma omp exit data [clauses]`
    - Fortran: `!$omp exit data [clauses]`


# Unstructured data

```c
class Vector {
    Vector(int n) : len(n) {
        v = new double[len];
        #pragma omp enter data alloc(v[0:len])
    }
    ~Vector() {
        #pragma omp exit data delete(v[0:len])
        delete[] v;
    }
    double v;
    int len;
};
```


# Enter data clauses

`map(alloc:var-list)`
  : `-`{.ghost}

- Allocate memory on the device

`map(to:var-list)`
  : `-`{.ghost}

- Allocate memory on the device and copy data from the host to the
  device


# Exit data clauses

`map(delete:var-list)`
  : `-`{.ghost}

- Deallocate memory on the device

`map(from:var-list)`
  : `-`{.ghost}

- Deallocate memory on the device and copy data from the device to the
  host


# Data directive: update

- Define variables to be updated within a data region between host and
  device memory
    - C/C++: `#pragma omp target update [clauses]`
    - Fortran: `!$omp target update [clauses]`
- Data transfer direction controlled by `map(to:var-list)` or
  `map(from:var-list)` clauses
    - direction from host perspective, *i.e.* `to` copies from host to device
- At least one data direction clause must be present



# Data directive: update

- `update` is a single line executable directive
- Useful for producing snapshots of the device variables on the host or
  for updating variables on the device
    - Pass variables to host for visualization
    - Communication with other devices on other computing nodes
- Often used in conjunction with
    - Asynchronous execution of OpenMP constructs
    - Unstructured data regions


# Example: update directive

<div class="column">
## C/C++

```c
float a[100];
int iter;
int maxit=100;

#pragma omp target data alloc(a) {
    /* Initialize data on device */
    init(a);
    for (iter=0; iter < maxit; iter++) {
        /* Computations on device */
        omp_compute(a);
        #pragma omp target update map(from:a) \
                if(iter % 10 == 0)
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

!$omp target data alloc(a)
    ! Initialize data on device
    call init(a)
    do iter=1,maxit
        ! Computations on device
        call omp_compute(a)
        !$omp target update map(from:a)
        !$omp& if(mod(iter,10)==0)
    end do
!$acc end data
```
</div>


# Data directive: declare target

- Makes a variable resident in accelerator memory
- Added at the declaration of a variable
- Data life-time on device is the implicit life-time of the variable
    - C/C++: `#pragma omp declare target [clauses]`
    - Fortran: `!$omp declare target [clauses]`


# Porting and unified memory

<div class="column">
- Porting a code with complicated data structures can be challenging
  because every field in type has to be copied explicitly
- Recent GPUs have *Unified Memory* and support for page faults
</div>

<div class="column">
```c
typedef struct points {
  double *x, *y;
  int n;
}

void process_points() {
  points p;

  #pragma omp target data map(alloc:p)
  {
    p.size = n;
    p.x = (double) malloc(...
    p.y = (double) malloc(...
    #pragma omp target update map(to:p)
    #pragma omp target update map(to:p.x[0:n],

    #pragma omp target
    // processs
```

</div>


# Unified memory

<div class="column">
- OpenMP 5.0 added a `requires` construct so that program can declare
  it assumes shared memory between host and device
- Compiler support in progress
</div>

<div class="column">
```c
typedef struct points {
  double *x, *y;
  int n;
}

void process_points() {
  points p;

  #pragma omp requires unified_shared_memory

  p.size = n;
  p.x = (double) malloc(...
  p.y = (double) malloc(...

  #pragma omp target
  // processs
```
</div>


# Summary

- Data directive
    - Structured data region
    - map types: `to`, `from`, `tofrom`, `alloc`, `delete`
- `enter data` and `exit data`
    - Unstructured data region
- Update directive
- Declare directive
