---
title:  OpenMP unstructured data regions and device functions
event:  CSC Summer School in High-Performance Computing 2022
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
  : Allocate memory on the device

`map(to:var-list)`
  : Allocate memory on the device and copy data from the host to the device


# Exit data clauses

`map(delete:var-list)`
  : Deallocate memory on the device

`map(from:var-list)`
  : Deallocate memory on the device and copy data from the device to the host


# Declare target

- Makes a variable resident in accelerator memory
- Added at the declaration of a variable
- Data life-time on device is the implicit life-time of the variable
    - C/C++: `#pragma omp declare target [clauses]`
    - Fortran: `!$omp declare target [clauses]`


# Porting and unified memory

<div class="column" style="width:43%">
- Porting a code with complicated data structures can be challenging
  because every field in type has to be copied explicitly
- Recent GPUs have *Unified Memory* and support for automatic data transfers
  with page faults
</div>

<div class="column" style="width:52%">
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
    #pragma omp target update map(to:p.x[0:n])
    ...
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
  ...
```
</div>


# Summary

- Implicit/explicit mapping
    - mapping types: `to`, `from`, `tofrom`, `alloc`, `delete`
- Structured data region
- Unstructured data region
    - `enter data` and `exit data`
- Data directives: update, reduction, declare target

# Device functions {.section}

# Function calls in compute regions

- Often it can be useful to call functions within loops to improve
  readability and modularisation
- By default OpenMP does not create accelerated regions for loops
  calling functions
- One has to instruct the compiler to compile a device version of the
  function


# Directive: `declare target`

- Define a function to be compiled for an accelerator as well as the host
- In C/C++  one puts `declare target` and `end declare target`
  around function declarations
- In Fortran one uses `!$declare target (name)` form
- The functions will now be compiled both for host and device execution


# Example: `declare target`

<div class="column">
**C/C++**
```c
#pragma omp declare target
void foo(float* v, int i, int n) {
    for ( int j=0; j<n; ++j) {
        v[i*n+j] = 1.0f/(i*j);
    }
}
#pragma omp end declare target

#pragma omp target teams loop
for (int i=0; i<n; ++i) {
    foo(v,i);  // executed on the device
}
```
</div>

<div class="column">
**Fortran**
```fortran
subroutine foo(v, i, n)
  !$omp declare target
  real :: v(:,:)
  integer :: i, n

  do j=1,n
     v(i,j,n) = 1.0/(i*j)
  enddo
end subroutine

!$omp target teams loop
do i=1,n
  call foo(v,i,n)
enddo
!$omp end target teams loop
```
</div>


# Summary

- Declare target directive
    - Enables one to write device functions that can be called within
      parallel loops
