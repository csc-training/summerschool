---
title:  OpenMP library routines and data sharing
event:  CSC Summer School in High-Performance Computing 2025
lang:   en
---


# OpenMP runtime library and environment variables {.section}

# OpenMP runtime library and environment variables

- OpenMP provides several means to interact with the execution
  environment. These operations include e.g.
    - setting the number of threads for parallel regions
    - requesting the number of CPUs
    - changing the default scheduling for work-sharing clauses
- Improves portability of OpenMP programs between different architectures
  (number of CPUs, etc.)


# Environment variables

- OpenMP standard defines a set of environment variables that all
  implementations have to support
- The environment variables are set before the program execution and they are
  read during program start-up
    - changing them during the execution has no effect
- We have already used `OMP_NUM_THREADS`


# Some useful environment variables

| Variable         | Action                                              |
|------------------|-----------------------------------------------------|
| OMP_NUM_THREADS  | Number of threads to use                            |
| OMP_PROC_BIND    | Bind threads to CPUs                                |
| OMP_PLACES       | Specify the bindings between threads and CPUs       |
| OMP_DISPLAY_ENV  | Print the current OpenMP environment info on stderr |


# Runtime functions

- Runtime functions can be used either to read the settings or to set
  (override) the values
- Function definitions are in
    - `omp.h` header file (C/C++)
    - `omp_lib` Fortran module (`omp_lib.h` header in some implementations)
- Two useful routines for finding out thread ID and number of threads:
    - `omp_get_thread_num()`
    - `omp_get_num_threads()`


# OpenMP conditional compilation

- In C/C++, one can use the `_OPENMP` macro to compile different code with and
  without OpenMP:

```c
#ifdef _OPENMP
    OpenMP specific code with, e.g., library calls
#else
    code without OpenMP
#endif
```


# Example: Hello world with OpenMP

<!-- Presentation suggestion: live coding, first without #ifdef
     (compilation without -fopenmp fails) and then with #ifdef
-->

<div class=column>
```fortranfree
program hello
  use omp_lib
  integer :: omp_rank
!$omp parallel
#ifdef _OPENMP
  omp_rank = omp_get_thread_num()
#else
  omp_rank = 0
#endif
  print *, 'Hello world! by &
        thread ', omp_rank
!$omp end parallel
end program hello
```
</div>

<div class=column>
```c
#include <stdio.h>
#include <omp.h>
int main(int argc, char* argv[]) {
  int omp_rank;
#pragma omp parallel
  {
#ifdef _OPENMP
    omp_rank = omp_get_thread_num();
#else
    omp_rank = 0;
#endif
    printf("Hello world! by thread %d\n",
           omp_rank);
  }
}
```
</div>



# Parallel regions and data sharing {.section}

# How do OpenMP threads interact?

- Because of the shared address space threads can interact using
  _shared variables_
- Threads often need some _private work space_ in addition to the shared
  variables
    - for example the index variable of a loop
- If threads write to the same shared variable a **race condition** appears
  <br>⇒ undefined end result
- Visibility of different variables is defined using _data-sharing clauses_
  in the parallel region definition


# Race condition in Hello world

<!-- Presentation suggestion: live coding, multiple runs with different outcome
     Note! Intel classic compiler optimizes somehow the race condition away
           with -O2 or higher, GNU and Clang-based compilers (including Intel
           OneAPI) show the race condition.
 -->

```c
#include <stdio.h>
#include <omp.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
  int omp_rank;
#pragma omp parallel
  {
    omp_rank = omp_get_thread_num();
    sleep(1);
    printf("Hello world! by thread %d\n", omp_rank);
  }
}
```
- All the threads write out the same thread number
- The result varies between successive runs


# omp parallel: data-sharing clauses

- **private(list)**
    - private variables are stored in the private stack of each thread
    - undefined initial value
    - undefined value after parallel region
- **firstprivate(list)**
    - same as *private*, but with an initial value that is the same as
      the original value defined outside the parallel region


# omp parallel: data-sharing clauses

- **shared(list)**
    - all threads can write to and read from a shared variable
- **default(private/shared/none)**
    - sets default for variables to be shared, private or not defined
    - in C/C++ default(private) is not allowed
    - default(none) can be useful for debugging as each variable has to be
      defined manually


# Default behaviour

- Most variables are _shared_ by default
    - global variables are shared among threads
        - C: static variables, file scope variables
        - Fortran: save and module variables, common blocks
        - `threadprivate(list)` can be used to make a private copy
- Private by default:
    - local variables of functions called from parallel region
    - variables declared within a block (C/C++)
    - outermost loop variables
- Good programming practice: declare all variables either shared or private


# Hello world without a race condition

```c
#include <stdio.h>
#include <omp.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
  int omp_rank;
#pragma omp parallel private(omp_rank)
  {
    omp_rank = omp_get_thread_num();
    sleep(1);
    printf("Hello world! by thread %d\n", omp_rank);
  }
}
```


# Data sharing example

<div class=column>
<small>main.c</small>
```c
int A[5];  // shared

int main(void) {
  int B[2];  // shared
  #pragma omp parallel
  {
    float c;  // private
    do_things(B);
    ...
  }
  return 0;
}
```
</div>
<div class=column>
<small>kernel.c</small>
```c
extern int A[5];  // shared

void do_things(int *var) {
  double wrk[10];  // private
  static int status; // shared
  ...
}
```
</div>


# Summary

- OpenMP runtime behavior can be controlled using environment variables
- OpenMP provides also library routines
- Visibility of variables in parallel region can be specified with
  data sharing clauses
    - **private** : each thread works with their own variable
    - **shared** : all threads can write to and read from a shared variable
- Race conditions possible when writing to shared variables
    - avoiding race conditions is key to correctly functioning OpenMP programs
