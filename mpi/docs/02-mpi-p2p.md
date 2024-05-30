---
title:  Point-to-point communication
event:  CSC Summer School in High-Performance Computing 2024
lang:   en
---

# Point-to-point communication{.section}

# Communication

<div class="column">

- Data is local to the MPI processes
    - They need to *communicate* to coordinate work
- Point-to-point communication
    - Messages are sent between two processes
- Collective communication
    - Involving a number of processes at the same time

</div>

<div class="column">

![](img/communication-schematic.png){.center width=50%}

</div>


# MPI point-to-point operations

- One process *sends* a message to another process that *receives* it with **`MPI_Send`** and **`MPI_Recv`** routines
- Sends and receives in a program should match – one receive per send

# MPI point-to-point operations

MPI_Send(`buffer`{.input}, `count`{.input}, `datatype`{.input}, `dest`{.input}, `tag`{.input}, `comm`{.input})
  : Performs a blocking send

MPI_Recv(`buffer`{.output}, `count`{.input}, `datatype`{.input}, `source`{.input}, `tag`{.input}, `comm`{.input}, `status`{.output})
  : Performs a blocking receive

<p>

- Each message (envelope) contains
    - The actual *data* (buffer) that is to be sent
    - The *number of elements* in the data
    - The *datatype* of each element of the data
    - The ranks of the *source* and *destination* processes
    - An identification number for the message (*tag*)
<p>
- Demo: `send_and_recv.c`


# Status parameter

- The status parameter in `MPI_Recv` contains information about the received data after the call has completed
  - The number of received elements
    - Use the function **`MPI_Get_count`(`status`{.input}, `datatype`{.input}, `count`{.output})**
    - Note that `count` parameter of `MPI_Recv` is the **maximum** number of elements to receive
  - The tag of the received message
    - C: `status.MPI_TAG`
    - Fortran 2008: `status%mpi_tag` (old Fortran `status(MPI_TAG)`)
  - The rank of the sender
    - C: `status.MPI_SOURCE`
    - Fortran 2008: `status%mpi_source` (old Fortran `status(MPI_SOURCE)`)


# "Buffers" in MPI

- The `buffer` arguments are memory addresses
- MPI assumes contiguous chunk of memory
    - `count` elements are send starting from the address
    - received elements are stored starting from the address
- In C/C++ `buffer` is pointer
    - For C++ `<array>` and `<vector>` containers, use `array.data()` method
- In Fortran arguments are passed by reference and variables can be passed as such to MPI calls
    - Note: be careful if passing non-contiguous array segmens such as <br>`a(1, 1:N)`


# MPI datatypes

- On a low level, MPI sends and receives stream of bytes
- MPI datatypes specify how the bytes should be interpreted
    - Allows data conversions in heterogenous environments (*e.g.* little endian to big endian)
- MPI has a number of predefined basic datatypes corresponding to C or Fortran datatypes
    - C examples: `MPI_INT` for `int` and `MPI_DOUBLE` for `double`
    - Fortran examples: `MPI_INTEGER` for `integer`, `MPI_DOUBLE_PRECISION` for `real64`
- Datatype `MPI_BYTE` for raw bytes is available both in C and Fortran
    - Portability can be an issue - be careful
- One can also define custom datatypes for communicating complex data


# MPI datatypes specific for C

| MPI type     |  C type       |
| ------------ | ------------- |
| `MPI_CHAR`   | `signed char` |
| `MPI_SHORT`  | `short int`   |
| `MPI_INT`    | `int`         |
| `MPI_LONG`   | `long int`    |
| `MPI_FLOAT`  | `float`       |
| `MPI_DOUBLE` | `double`      |


# MPI datatypes specific for Fortran

| MPI type               |  Fortran type    |
| ---------------------- | ---------------- |
| `MPI_CHARACTER`        | character        |
| `MPI_INTEGER`          | integer          |
| `MPI_REAL`             | real32           |
| `MPI_DOUBLE_PRECISION` | real64           |
| `MPI_COMPLEX`          | complex          |
| `MPI_DOUBLE_COMPLEX`   | double complex   |
| `MPI_LOGICAL`          | logical          |


# Case study: parallel sum on two processes

<div class=column>
![](img/case_study_left-01.png){.center width=45%}
</div>

<div class=column>
- Array initially on process #0 (P0)
- Parallel algorithm:
    1. **Scatter**:
    P0 sends half of the array to process P1

    2. **Compute**:
    P0 & P1 sum independently their segments

    3. **Reduction**:
    Partial sum on P1 is sent to P0 and
    P0 sums the partial sums

</div>

# Case study: parallel sum on two processes

<div class=column>
![](img/case_study_left-03.png){.center width=45%}
</div>
<div class=column>
**Step 1**: Scatter array
<p>
![](img/case_study_right-02.png){.center width=90%}
</div>

# Case study: parallel sum on two processes

<div class=column>
![](img/case_study_left-04.png){.center width=45%}
</div>
<div class=column>
**Step 2**: Compute the sum in parallel
<p>
![](img/case_study_right-03.png){.center width=90%}
</div>

# Case study: parallel sum on two processes

<div class=column>
![](img/case_study_left-06.png){.center width=45%}
</div>
<div class=column>
**Step 3.1**: Gather partial sums
<p>
![](img/case_study_right-05.png){.center width=90%}
</div>

# Case study: parallel sum on two processes

<div class=column>
![](img/case_study_left-07.png){.center width=45%}
</div>
<div class=column>
**Step 3.2**: Compute the total sum
<p>
![](img/case_study_right-06.png){.center width=90%}
</div>


# Demo

- `parallel_sum.c`


# Blocking routines and deadlocks

- `MPI_Send` and `MPI_Recv` are blocking routines
    - `MPI_Send` exits once the send buffer can be safely read and written to
    - `MPI_Recv` exits once it has received the message in the receive buffer
- Completion depends on other processes → risk for *deadlocks*
    - For example, all processes are waiting in `MPI_Recv` but no-one is sending <br>
      → the program is stuck forever (deadlock)


# Summary

- Point-to-point communication = messages are sent between two MPI processes
- Point-to-point operations enable any parallel communication pattern (in principle)
  - `MPI_Send` and `MPI_Recv`
- Status parameter of `MPI_Recv` contains information about the message after the receive is completed
- `MPI_Send` and `MPI_Recv` are blocking routines
  - Beware of deadlocks

