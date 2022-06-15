---
title:  Introduction to OpenMP
author: CSC Training
date:   2022
lang:   en
---

# OpenMP {.section}


# What is OpenMP?

- A collection of _compiler directives_ and _library routines_ ,
  together with a _runtime system_, for
  **multi-threaded**, **shared-memory parallelization**
- Fortran 77/9X/0X and C/C++ are supported
- Latest version of the standard is 5.2 (November 2021)
    - Full support for accelerators (GPUs)
    - Support latest versions of C, C++ and Fortran
    - Support for a fully descriptive loop construct
    - and more
- Compiler support for 5.0 is still incomplete
- This course discusses mostly features present in < 4.5


# Why would you want to learn OpenMP?

- OpenMP parallelized program can be run on your many-core workstation or on a
  node of a cluster
- Enables one to parallelize one part of the program at a time
    - Get some speedup with a limited investment in time
    - Efficient and well scaling code still requires effort
- Serial and OpenMP versions can easily coexist
- Hybrid MPI+OpenMP programming


# Three components of OpenMP

- Compiler directives, i.e. language extensions
    - Expresses shared memory parallelization
    - Preceded by sentinel, can compile serial version

- Runtime library routines
    - Small number of library functions
    - Can be discarded in serial version via conditional compiling

- Environment variables
    - Specify the number of threads, thread affinity etc.


# OpenMP directives

- OpenMP directives consist of a *sentinel*, followed by the directive
  name and optional clauses
- C/C++:
```C
#pragma omp directive [clauses]
```
- Fortran:
```Fortran
!$omp directive [clauses]
```

- Directives are ignored when code is compiled without OpenMP support


# Compiling an OpenMP program

- Compilers that support OpenMP usually require an option (flag) for enabling it
    - Most compilers (GNU, Intel, Cray) nowadays support `-fopenmp`
    - Intel legacy foption: `-qopenmp`
    - NVIDIA: `-mp`


# Parallel construct

<div class="column">

- Defines a *parallel region*
    - C/C++:
    ```C
    #pragma omp parallel [clauses]
       structured block
    ```
    - Fortran:
    ```fortran
    !$omp parallel [clauses]
       structured block
    !$omp end parallel
    ```
- Prior to it only one thread (main)
- Creates a team of threads
- Barrier at the end of the block

</div>
<div class="column">

- SPMD: Single Program Multiple Data
![](img/omp-parallel.png){.center width=60%}

</div>


# Example: "Hello world" with OpenMP

<div class="column">
```fortran
program omp_hello

   write(*,*) "Hello world! -main"
!$omp parallel
   write(*,*) ".. worker reporting for duty."
!$omp end parallel
   write(*,*) "Over and out! -main"

end program omp_hello
```
```bash
> gfortran -fopenmp omp_hello.F90 -o omp
> OMP_NUM_THREADS=3 ./omp
 Hello world! -main
 .. worker reporting for duty.
 .. worker reporting for duty.
 .. worker reporting for duty.
 Over and out! -main
```
</div>

<div class="column">
```c
#include <stdio.h>

int main(int argc, char* argv[]) {
  printf("Hello world! -main\n");
#pragma omp parallel
  {
    printf(".. worker reporting for duty.\n");
  }
  printf("Over and out! -main\n");
}
```
```bash
> gcc -fopenmp omp_hello.c -o omp
> OMP_NUM_THREADS=3 ./omp
Hello world! -main
.. worker reporting for duty.
.. worker reporting for duty.
.. worker reporting for duty.
Over and out! -main
```
</div>


# How to distribute work?

- Each thread executes the same code within the parallel region
- OpenMP provides several constructs for controlling work distribution
    - for/do construct
    - single/master/masked construct
    - sections construct
    - task construct
    - workshare construct (Fortran only)
- Thread ID can be queried and used for distributing work manually
  (similar to MPI rank)

# for/do construct

- Directive instructing compiler to share the work of a loop
    - Each thread executes only part of the loop

```C
#pragma omp for [clauses]
...
```
```fortran
!$omp do [clauses]
...
!$omp end do
```
- in C/C++ limited only to "canonical" for-loops. Iterator base loops are
  also possible in C++
- Construct must reside inside a parallel region
    - Combined construct with omp parallel: \
          `#pragma omp parallel for` / `!$omp parallel do`


# for/do construct

```fortran
!$omp parallel
!$omp do
  do i = 1, n
     z(i) = x(i) + y(i)
  end do
!$omp end do
!$omp end parallel
```

```c
#pragma omp parallel
{
  #pragma omp for
  for (i=0; i < n; i++)
    z[i] = x[i] + y[i];
}
```

# Workshare directive (Fortran only)

- In Fortran many array operations can be done conveniently with array
  syntax, *i.e.* without explicit loops
    - Array assignments, forall and where statements *etc.*
- The `workshare` directive divides the execution of the enclosed structured
  block into separate units of work, each of which is executed only once

```fortran
real :: a(n,n), b(n,n), c(n,n) d(n,n)
...
!$omp parallel
  !$omp workshare
     c = a * b
     d = a + b
  !$omp end workshare
!$omp end parallel
```

# Summary

- OpenMP can be used with compiler directives
    - Ignored when code is build without OpenMP
- Threads are launched within **parallel** regions
- Simple loops can be parallelized easily with a `for`/`do` construct
