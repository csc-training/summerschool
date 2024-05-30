---
title:  Introduction to Message Passing Interface (MPI)
event:  CSC Summer School in High-Performance Computing 2024
lang:   en
---

# Message passing interface

- MPI is an application programming interface (API) for distributed parallel
  computing
- MPI programs are portable and scalable
    - the same program can run on different types of computers, from
      laptops to supercomputers
- MPI is flexible and comprehensive
    - large (hundreds of procedures)
    - concise (only 10-20 procedures are typically needed)

# MPI standard and implementations

- MPI is a standard, first version (1.0) published in 1994, latest (4.1) in 2023
    - <https://www.mpi-forum.org/docs/>
- The MPI standard is implemented by different MPI implementations
    - [OpenMPI](http://www.open-mpi.org/)
    - [MPICH](https://www.mpich.org/)
    - [Intel MPI](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/mpi-library.html)


# Processes and threads

![](img/processes-threads-highlight-proc.png){.center width=85%}

<div class="column">

**Process**

- Independent execution units
- Have their own state information and *own memory* address space

</div>

<div class="column">

**Thread**

- A single process may contain multiple threads
- Have their own state information, but *share* the *same memory*
  address space

</div>



# Execution model in MPI

- Normally, parallel program is launched as a set of *independent*, *identical
  processes*
    - execute the *same program code* and instructions
    - processes can reside in different nodes (or even in different computers)
- The way to launch parallel program depends on the computing system
    - **`mpiexec`**, **`mpirun`**, **`srun`**, **`aprun`**, ...
    - **`srun`** on LUMI, Mahti, and Puhti
- MPI supports also dynamic spawning of processes and launching *different*
  programs communicating with each other
    - not covered here

# MPI ranks

<div class="column">
- MPI runtime assigns each process a unique rank (index)
    - identification of the processes
    - ranks range from 0 to N-1
- Processes can perform different tasks and handle different data based
  on their rank
</div>
<div class="column">
```c
double a;
if (rank == 0) {
   a = 1.0;
   ...
}
else if (rank == 1) {
   a = 0.7;
   ...
}
...
```
</div>

# Data model

- All variables and data structures are local to the process
- Processes can exchange data by sending and receiving messages

![](img/data-model.png){.center width=100%}


# MPI library

- Information about the communication framework
    - the number of processes
    - the rank of the process
- Communication between processes
    - sending and receiving messages between two or several processes
- Synchronization between processes
- Advanced features
    - Communicator manipulation, user defined datatypes, one-sided communication, ...


# MPI communicator

- Communicator is an object connecting a group of processes
    - It defines the communication framework
- Most MPI functions require a communicator as an argument
- Initially, there is always a communicator **`MPI_COMM_WORLD`** which
  contains all the processes
- Users can define custom communicators


# Interlude

Message passing game


# Programming MPI

- The MPI standard defines interfaces to C and Fortran programming
  languages
    - No C++ bindings in the standard, C++ programs use the C interface
    - There are unofficial bindings to Python, Rust, R, ...
- Call convention in C (*case sensitive*):<br>
`error_code = MPI_Xxxx(parameter, ...)`
    - Return value is the error code (e.g., `MPI_SUCCESS`)
    - Some arguments have to be passed as pointers
- Call convention in Fortran (*case insensitive*):<br>
`call mpi_xxxx(parameter, ..., error_code)`
    - Error code in the last argument


# Writing an MPI program

<div class="column">
## C:

```c
// Include MPI header file
#include <mpi.h>

int main(int argc, char *argv[])
{
    // Start by calling MPI_Init()
    MPI_Init(&argc, &argv);

    ...  // program code

    // Call MPI_Finalize() before exiting
    MPI_Finalize();
}


```
</div>

<div class="column">
## Fortran 2008:

```fortranfree
program my_prog
  ! Use MPI module
  use mpi_f08
  implicit none
  integer :: rc

  ! Start by calling mpi_init()
  call mpi_init(rc)

  ... ! program code

  ! Call mpi_finalize() before exiting
  call mpi_finalize(rc)
end program my_prog
```
</div>

- Older Fortran codes might have `use mpi` or `include 'mpif.h'`

# Compiling an MPI program

- MPI is a library (+ runtime system)
- In principle, MPI programs can be built with standard compilers
  (`gcc` / `g++` / `gfortran`) with the appropriate `-I` / `-L` / `-l`
  options
- Most MPI implementations provide convenience wrappers, typically
  `mpicc` / `mpicxx` / `mpif90`, for easier building
    - MPI-related options are automatically included

```bash
mpicc -o my_mpi_prog my_mpi_code.c
mpicxx -o my_mpi_prog my_mpi_code.cpp
mpif90 -o my_mpi_prog my_mpi_code.F90
```

# Compiling an MPI program on LUMI

- On LUMI (HPE Cray EX), there are `cc` / `CC` / `ftn` compiler wrappers
  invoking the correct compiler
  - Use these instead of the `mpi*` wrappers

```bash
cc -o my_mpi_prog my_mpi_code.c
CC -o my_mpi_prog my_mpi_code.cpp
ftn -o my_mpi_prog my_mpi_code.F90
```

# MPI calls

- Syntax on slides: **`MPI_Function(` `input_arg`{.input} `, ` `output_arg`{.output} `)`**
  - Input arguments in `red`{.input} and output arguments in `blue`{.output}
- Note that in C the output arguments are always pointers!
- See references of MPI implementations for detailed function definitions:
  - <https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man3/index.html>
  - <https://www.mpich.org/static/docs/v4.2.x/>
  - `man MPI_Function` on LUMI (e.g., `man MPI_Init`)


# First five MPI commands: Initialization and finalization

MPI_Init(...)
  : Initializes the MPI execution environment

MPI_Finalize()
  : Terminates the MPI execution environment

<p>
- Demo: `hello.c`

# First five MPI commands: Information about the communicator

MPI_Comm_size(`comm`{.input}, `size`{.output})
  : Determines the size of the group associated with a communicator

MPI_Comm_rank(`comm`{.input}, `rank`{.output})
  : Determines the rank of the calling process in the communicator

<p>
- Demo: `hello_rank.c`

# First five MPI commands: Synchronization

MPI_Barrier(`comm`{.input})
  : Waits until all ranks within the communicator reaches the call

<p>
- Demo: `barrier.c`

# Summary

- In parallel programming with MPI, the key concept is a set of independent processes
- Data is always local to the process
- Processes can exchange data by sending and receiving messages
- MPI defines functions for communication and synchronization between processes

# Web resources

- List of MPI functions with detailed descriptions
    - <https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man3/index.html>
    - <https://www.mpich.org/static/docs/v4.2.x/>
    - <https://rookiehpc.org/mpi/docs/>
- Good online MPI tutorials
    - <https://hpc-tutorials.llnl.gov/mpi/>
    - <http://mpitutorial.com/tutorials/>
    - <https://www.youtube.com/watch?v=BPSgXQ9aUXY>
- MPI coding game in C
    - <https://www.codingame.com/playgrounds/47058/have-fun-with-mpi-in-c/lets-start-to-have-fun-with-mpi>
