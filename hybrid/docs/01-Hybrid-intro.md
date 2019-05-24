---
title:  OpenMP
author: CSC Summerschool
date:   2019-06
lang:   en
---

# Basic concepts: Processes and threads {.section}


# Processes and threads

![](img/processes-threads.png)

<div class="column">

## Process

- Independent execution units
- Have their own state information and *own memory* address space

</div>

<div class="column">

## Thread

- A single process may contain multiple threads
- Have their own state information, but *share* the *same memory*
  address space


</div>


# Processes and threads

![](img/processes-threads.png)


<div class="column">
## Process

- Long-lived: spawned when parallel program started, killed when
  program is finished
- Explicit communication between processes

**MPI**

</div>

<div class="column">

## Thread

- Short-lived: created when entering a parallel region, destroyed
  (joined) when region ends
- Communication through shared memory

**OpenMP**

</div>


# What is OpenMP?

* A collection of _compiler directives_ and _library routines_ for **multi-threaded**, **shared-memory parallelization**
* Fortran 77/9X/03 and C/C++ are supported
* Latest version of the standard is 4.5 (November 2015)
	- support for attached devices
	- support for thread affinity

# Why would you want to learn OpenMP?

* OpenMP parallelized program can be run on your many-core workstation or on a node of a cluster
* Enables one to parallelize one part of the program at a time
	- Get some speedup with a limited investment in time
	- Efficient and well scaling code still requires effort
* Serial and OpenMP versions can easily coexist
* Hybrid MPI+OpenMP programming

# Three components of OpenMP 

* Compiler directives, i.e. language extensions, for shared memory parallelization

|`   `	| directive     | construct     |clauses	  |
|-----	|-----------	|-------------  |-----		  |
|C/C++  | `#pragma omp` | `parallel` 	| `shared(data)`  |
|Fortran| `!$omp`      	| `parallel`    | `shared(data)`  |

* Runtime library routines (Intel: libiomp5, GNU: libgomp)
	- Conditional compilation to build serial version
* Environment variables
	- Specify the number of threads, thread affinity etc.

# OpenMP directives 

* Sentinels precede each OpenMP directive
	- C/C++: 	`#pragma omp`
	- Fortran:	`!$omp`
* Conditional compilation with `_OPENMP` macro:
```c
#ifdef _OPENMP
	OpenMP specific code
#else
	Code without OpenMP
#endif
```
# Compiling an OpenMP program

* Compilers that support OpenMP usually require an option that enables the feature
	- GNU: `-fopenmp`
	- Intel: `-qopenmp`
	- Cray: `-h omp`
		* OpenMP enabled by default, -h noomp disables
	- PGI: `-mp[=nonuma,align,allcores,bind]`
	- Without these options a serial version is compiled!

# Example: Helloworld with OpenMP

<div class=column>
```fortran
program hello
  use omp_lib
  integer :: omp_rank
!$omp parallel private(omp_rank)
  omp_rank = omp_get_thread_num()
  print *, 'Hello world! by &
	    thread ', omp_rank
!$omp end parallel
end program hello
```
```bash
> ftn –h omp omp_hello.f90 -o omp
> aprun –e OMP_NUM_THREADS=4 -n 1 -d 4 ./omp
 Hello world! by thread             0
 Hello world! by thread             2
 Hello world! by thread             3
 Hello world! by thread             1
```

</div>
<div class=column>
```c
#include <stdio.h>
#include <omp.h>
int main(int argc, char argv[]){
  int omp_rank;
#pragma omp parallel private(omp_rank){
  omp_rank = omp_get_thread_num();
  printf("Hello world! by 
  	thread %d", omp_rank);}
  }
```
```bash
> cc –h omp omp_hello.c -o omp
> aprun –e OMP_NUM_THREADS=4 -n 1 -d 4 ./omp
 Hello world! by thread             2
 Hello world! by thread             3
 Hello world! by thread             0
 Hello world! by thread             1
```
</div>

# Parallel regions and data sharing {.section}

# Parallel construct

<div class=column>
* Defines a parallel region  
	- C/C++:   
	`#pragma omp parallel [clauses]`
	- Fortran:   
	`!$omp parallel [clauses]`
	- Prior to it only one thread, master
	- Creates a team of threads: master+slave threads
	- At end of the block is a barrier and all shared data is synchronized
</div>
<div class=column>
SPMD: Single Program Multiple Data
![](img/omp-parallel.png)

</div>

# How do the threads interact?

* Because of the shared address space threads can interact using _shared variables_
* Threads often need some _private work space_ together with shared variables
	- for example the index variable of a loop
* Visibility of different variables is defined using _data-sharing clauses_ in the parallel region definition

# omp parallel: data-sharing clauses

* **private(list)**  
	- Private variables are stored in the  private stack of each thread
	- Undefined initial value
	- Undefined value after parallel region
* **firstprivate(list)**  
	- Same as private variable, but with an initial value that is the same as the original objects defined outside the parallel region

# omp parallel: data-sharing clauses

* **shared(list)**
	- All threads can write to, and read from 
	- a shared variable
	- Variables are shared by default
* **default(private/shared/none)**
	- Sets default for variables to be shared, private or not defined
	- In C/C++ default(private) is not allowed
	- default(none) can be useful for debugging as each variable has to be defined manually  

_Race condition = 
a thread accesses a 
variable while another 
writes into it_

# Default behaviour

* Most variables are _shared_ by default
	- Global variables are shared among threads
		* C: static variables, file scope variables
		* Fortran: save and module variables, common blocks
		* `threadprivate(list)` can be used to make a private copy
* Private by default:
	- Stack variables of functions called from parallel region
	- Automatic variables within a block

# Data sharing example

<div class=column>
main.c
```c
int A[5];

int main(void) {
    int B[2];
#pragma omp parallel
{
    float c;
    do_things(B);
    ...
}
    return 0;
}
```
</div>
<div class=column>
kernel.c
```c
extern int A[5]; 

void do_things(int *var) {
    double wrk[10];
    static int status;
    ...
}
```
</div>
