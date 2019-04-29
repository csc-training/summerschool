---
title:  Message-Passing Interface (MPI) 
author: CSC Summerschool
date:   2019-06
lang:   en
---

# Point-to-point communication{.section}

# Communication

<div class="column">

- Data is local to the MPI processes
    - they need to *communicate* to coordinate work
- Point-to-point communication
    - Messages are sent between two processes
- Collective communication
    - Involving a number of processes at the same time

</div>

<div class="column">

![](communication-schematic.svg){.center width=50%}

</div>


# MPI point-to-point operations

- One process *sends* a message to another process that *receives* it with
  `MPI_Send` and `MPI_Recv` routines
- Sends and receives in a program should match – one receive per send
- Each message (envelope) contains
    - The actual *data* that is to be sent
    - The *datatype* of each element of data
    - The *number of elements* the data consists of
    - An identification number for the message (*tag*)
    - The ranks of the *source* and *destination* process
