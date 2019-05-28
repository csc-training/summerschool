---
title:  Message-Passing Interface (MPI) 
author: CSC Summerschool
date:   2019-06
lang:   en
---

# Processes and threads

![](images/processes-threads.svg){.center width=100%}

<div class="column">

<br>

## Process

- Independent execution units
- Have their own state information and *own memory* address space

</div>

<div class="column">

<br>

## Thread

- A single process may contain multiple threads
- Have their own state information, but *share* the *same memory*
  address space

</div>


# Processes and threads

![](images/processes-threads.svg){.center width=100%}


<div class="column">

<br>

## Process

- Long-lived: spawned when parallel program started, killed when
  program is finished
- Explicit communication between processes

</div>

<div class="column">

<br>

## Thread

- Short-lived: created when entering a parallel region, destroyed
  (joined) when region ends
- Communication through shared memory

</div>

# Getting started with MPI {.section}

# Message-passing interface

- MPI is an application programming interface (API) for communication
  between separate processes
    - The most popular *distributed* parallel computing method
    - MPI programs are portable and scalable
- MPI is flexible and comprehensive
    - Several communication methods and patterns
    - Parallel IO
- MPI standardization by MPI Forum
    - Latest version is 3.1, version 1.0 in 1994

# Execution model in MPI

- Parallel program is launched as set of *independent*, *identical
  processes*
    - The same program code and instructions
- MPI runtime assigns each process a *rank*
    - identification of the processes
    - Processes can perform different tasks and handle different data
      basing on their rank
    - Can reside in different nodes
- The way to launch parallel program is implementation dependent

# MPI ranks

- MPI runtime assigns each process a unique rank
	- identification of the processes
	- ranks start from 0 and extend to N-1
- Processes can perform different tasks and handle different data basing on their rank

```c
if (rank == 0) {
   ...
   }
if (rank == 1) {
   ...
   }
```

# Data model

- All variables and data structures are local to the process
- Processes can exchange data by sending and receiving messages

# The MPI library

- Information about the communication framework
    - number of processes
	- rank of the process
- Communication between processes
	- sending and receiving messages between two processes
	- sending and receiving messages between several processes
- Synchronization between processes
- Advanced features

# Programming MPI

- The MPI standard defines interfaces to C and Fortran programming languages
	- There are unofficial bindings to eg. Python, Perl and Java
- C call convention  *Case sensitive*   
`rc = MPI_Xxxx(parameter,...)`
	- some arguments have to be passed as pointers
- Fortran call convention *Case insensitive*  
`call mpi_xxxx(parameter,..., rc)`
	* return code in the last argument<Paste>

# Writing an MPI program

- Include the MPI header files   
C`---> #include <mpi.h>`   
Fortran`---> use mpi`
- Start by calling the routine **MPI_Init**
- Write the program
- Call **MPI_Finalize** before exiting from the main program

# Presenting syntax

![](images/presenting_syntax.svg){.center width=90%}

# First five MPI commands

<div class=column>
- Set up the MPI environment

**`MPI_Init()`**

<br>

- Finalize MPI environment

**`MPI_Finalize()`**

- Synchronize processes

MPI_Barrier(`comm`{.input})
  : `comm`{.input} 
    : communicator
</div>
<div class=column>
- Synchronize processes

MPI_Barrier(`comm`{.input})
  : `comm`{.input} 
    : communicator
</div>

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

# MPI communicator

- Communicator is an object connecting a group of processes, i.e. the
  communication framework
- Most MPI functions require communicator as an argument, i.e. take
  place in a given context
- Initially, there is always a communicator **MPI_COMM_WORLD** which
  contains all the processes and **MPI_COMM_SELF** that includes only
  the calling rank
- Users can define custom communicators

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
