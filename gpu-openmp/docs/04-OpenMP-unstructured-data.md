---
title:  OpenMP unstructured data regions
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

<div class="column">
- Porting a code with complicated data structures can be challenging
  because every field in type has to be copied explicitly
- Recent GPUs have *Unified Memory* and support for automatic data transfers
  with page faults
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
