---
title:  Message-Passing Interface (MPI) 
author: CSC Summerschool
date:   2019
lang:   en
---

# Agenda {.table-grid}

<small>

<div class=column>

| Friday         |                               | 
| -------------: |------------------------------ | 
|  9:00 - 10:00  | Introduction to MPI           |
| 10:00 - 10:15  | Coffee break                  |
| 10:15 - 11:00  | Point-to-point communication  |
| 11:00 - 12:00  | Exercises                     |
| 12:00 - 13:00  | Lunch                         |
| 13:00 - 13:45  | Collective communication      |
| 13:45 - 14:30  | Exercises                     |
| 14:30 - 14:45  | Coffee break                  |
| 14:45 - 15:00  | User defined communicators    |
| 15:00 - 15:45  | Exercises                     | 

</div>

<div class=column>

| Saturday       |                            | 
| -------------: | -------------------------- | 
|  9:00 - 10:00  | User defined datatypes 1   |
| 10:00 - 10:15  | Coffee break               |
| 10:15 - 11:00  | User defined datatypes 2   |
| 11:00 - 12:00  | Exercises                  |
| 12:00 - 13:00  | Lunch                      |
| 13:00 - 13:30  | Non-blocking communication |
| 13:30 - 14:30  | Exercises                  |
| 14:30 - 14:45  | Coffee break               |

</div>

</small>

# Basic concepts in MPI {.section}

# Message-passing interface

- MPI is an application programming interface (API) for distributed parallel 
  computing
- MPI programs are portable and scalable
    - the same program can run on different types of computers, from 
      laptops to supercomputers
- MPI is flexible and comprehensive 
    - large (hundreds of procedures)
    - concise (only 10-20 procedures are typically needed)

# Processes and threads

![](images/processes-threads-highlight-proc.svg){.center width=90%}

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



# Execution model in MPI

- Parallel program is launched as set of *independent*, *identical
  processes*
    - execute the *same program code* and instructions
    - processes can reside in different nodes (or even in different computers)
- The way to launch parallel program depends on the computing system
    - **`mpiexec`**, **`mpirun`**, **`aprun`**, **`srun`**, ...
    - **`aprun`** on sisu.csc.fi, **`srun`** on taito.csc.fi

# MPI ranks

<div class="column">
- MPI runtime assigns each process a unique rank
    - identification of the processes
    - ranks start from 0 and extend to N-1
- Processes can perform different tasks and handle different data based 
  on their rank
</div>
<div class="column">
```c
if (rank == 0) {
   ...
   }
if (rank == 1) {
   ...
   }
```
</div>

# Data model

- All variables and data structures are local to the process
- Processes can exchange data by sending and receiving messages

![](images/data-model.svg){.center width=100%}

# The MPI library

- Information about the communication framework
    - number of processes
    - rank of the process
- Communication between processes
    - sending and receiving messages between two processes
    - sending and receiving messages between several processes
- Synchronization between processes
- Advanced features

# MPI communicator

- Communicator is an object connecting a group of processes, i.e. the
  communication framework
- Most MPI functions require communicator as an argument
- Initially, there is always a communicator **MPI_COMM_WORLD** which
  contains all the processes
- Users can define custom communicators

# Programming MPI

- The MPI standard defines interfaces to C and Fortran programming languages
	- There are unofficial bindings to eg. Python, Perl and Java
- C call convention (*case sensitive*)<br>
`rc = MPI_Xxxx(parameter,...)`
    - some arguments have to be passed as pointers
- Fortran call convention (*case insensitive*)<br>
`call mpi_xxxx(parameter,..., rc)`
    - return code in the last argument

# Writing an MPI program

- C: include the MPI header file
```c
#include <mpi.h>
```
- Fortran: use MPI module
```fortran
use mpi_f08
```
(older Fortran codes might have `use mpi` or `include 'mpif.h'`)

- Start by calling the routine **MPI_Init**
- Write the program
- Call **MPI_Finalize** before exiting from the main program

# Presenting syntax

- MPI calls are presented as pseudocode
    - actual C and Fortran interfaces are given in reference section
    - Fortran error code argument not included

MPI_Function(`arg1`{.input}, `arg2`{.output})
  : `arg1`{.input}
    : input arguments in red
  : `arg2`{.output}
    : output arguments in blue. Note that in C the output arguments are always
      pointers to a variable


# First five MPI commands

- Initialization and finalization

MPI_Init
  : (in C `argc`{.input} and `argv`{.input} pointer arguments are needed)

MPI_Finalize
  : 

# First five MPI commands

- Information about the communicator

MPI_Comm_size(`comm`{.input}, `size`{.output})
  : `comm`{.input}
    : communicator
  : `size`{.output}
    : number of processes in the communicator

MPI_Comm_rank(`comm`{.input}, `rank`{.output})
  : `comm`{.input}
    : communicator
  : `rank`{.output}
    : rank of this process

# First five MPI commands

- Synchronization between processes
    - wait until everybody within the communicator reaches the call 

MPI_Barrier(`comm`{.input})
  : `comm`{.input}
    : communicator


# Summary 

- In parallel programming with MPI, the key concept is a set of
  independent processes
- Data is always local to the process
- Processes can exchange data by sending and receiving messages
- The MPI library contains functions for communication and
  synchronization between processes

# Web resources 

- List of MPI functions with detailed descriptions  
<http://mpi.deino.net/mpi_functions/index.htm>
- Good online MPI tutorials   
<https://computing.llnl.gov/tutorials/mpi>  
<http://mpitutorial.com/tutorials/>
- MPI 3.1 standard <http://www.mpi-forum.org/docs/>
- MPI implementations   
	* MPICH2 <http://www.mcs.anl.gov/research/projects/mpich2/>
	* OpenMPI <http://www.open-mpi.org/>
