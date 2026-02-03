<!--
SPDX-FileCopyrightText: 2010 CSC - IT Center for Science Ltd. <www.csc.fi>

SPDX-License-Identifier: CC-BY-4.0
-->

---
title:  OpenMP data sharing
event:  CSC Summer School in High-Performance Computing 2026
lang:   en
---

# Outline

- Data sharing in OpenMP


# Data sharing {.section}

# How do OpenMP threads interact?

- Threads share the memory address space, so they can interact using _shared variables_
- Threads often need some _private variables_ in addition to the shared variables
  - For example the index variable of a multithreaded loop
- If multiple threads write to the same shared variable a **race condition** appears
  - End results varies randomly depending on the order in which the threads executed
  - <https://deadlockempire.github.io/>
- Visibility of different variables is defined using _data-sharing clauses_


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

# Summary {.section}

# Summary

- Visibility of variables in parallel region can be specified with
  data sharing clauses
    - **private** : each thread works with their own variable
    - **shared** : all threads can write to and read from a shared variable
- Race conditions possible when writing to shared variables
    - avoiding race conditions is key to correctly functioning OpenMP programs

