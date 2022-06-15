---
title:  OpenMP declare GPU functions
event:  CSC Summer School in High-Performance Computing 2022
lang:   en
---


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
## C/C++
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
## Fortran
```fortran
subroutine foo(v, i, n)
  !$omp declare target
  real :: v(:,:)
  integer :: i, n

  do j=1,n
     v(i,j) = 1.0/(i*j)
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
