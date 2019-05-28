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
    - They need to *communicate* to coordinate work
- Point-to-point communication
    - Messages are sent between two processes
- Collective communication
    - Involving a number of processes at the same time

</div>

<div class="column">

![](images/communication-schematic.svg){.center width=50%}

</div>


# MPI point-to-point operations

- One process *sends* a message to another process that *receives* it
  with `MPI_Send` and `MPI_Recv` routines
- Sends and receives in a program should match – one receive per send
- Each message (envelope) contains
    - The actual *data* that is to be sent
    - The *datatype* of each element of data
    - The *number of elements* the data consists of
    - An identification number for the message (*tag*)
    - The ranks of the *source* and *destination* process

# Case study: parallel sum 

<div class=column>
![](images/case_study_left-01.svg){.center width=45%}
</div>

<div class=column>
- Array initially on process #0 (P0)
- Parallel algorithm
	* **Scatter**  
	Half of the array is sent to process 1

	* **Compute**  
	P0 & P1 sum independently their segments

	* **Reduction**  
	Partial sum on P1 sent to P0 
	P0 sums the partial sums

</div>

# Case study: parallel sum 

<div class=column>
![](images/case_study_left-02.svg){.center width=45%}
</div>
<div class=coulumn>

**Step 1.1**: Receive call in scatter

<p>
![](images/case_study_right-01.svg){.center width=50%}
<p>
P1 issues MPI_Recv to receive half of the array from P0
</div>


# Case study: parallel sum 

<div class=column>
![](images/case_study_left-03.svg){.center width=45%}
</div>
<div class=coulumn>
**Step 1.2**: Send call in scatter

<p>
![](images/case_study_right-02.svg){.center width=50%}
<p>
P0 issues an MPI_Send to send the lower part of the array to P1
</div>

# Case study: parallel sum 

<div class=column>
![](images/case_study_left-04.svg){.center width=45%}
</div>
<div class=coulumn>
**Step 2**: Compute the sum in parallel

<p>
![](images/case_study_right-03.svg){.center width=50%}
<p>
Both P0 & P1 compute their partial sums and store them locally
</div>

# Case study: parallel sum 

<div class=column>
![](images/case_study_left-05.svg){.center width=45%}
</div>
<div class=coulumn>
**Step 3.1**: Receive call in reduction

<p>
![](images/case_study_right-04.svg){.center width=50%}
<p>
P0 issues an MPI_Recv operation for receiving P1’s partial sum
</div>

# Case study: parallel sum 

<div class=column>
![](images/case_study_left-06.svg){.center width=45%}
</div>
<div class=coulumn>
**Step 3.2**: Send call in reduction

<p>
![](images/case_study_right-05.svg){.center width=50%}
<p>
P1 sends the partial sum to P0
</div>

# Case study: parallel sum 

<div class=column>
![](images/case_study_left-07.svg){.center width=45%}
</div>
<div class=coulumn>
**Step 3.3**: compute the final answer

<p>
![](images/case_study_right-06.svg){.center width=50%}
<p>
P0 computes the total sum
</div>

# Send operation {.split-definition}

MPI_Send(`buffer`{.input}, `count`{.input}, `datatype`{.input}, `dest`{.input}, `tag`{.input}, `comm`{.input})
  : `buffer`{.input}
    : The data to be sent
    
    `count`{.input}
    : Number of elements in buffer

    `datatype`{.input}
    : Type of elements in buffer (see later slides)

    `dest`{.input}
    : The rank of the receiver

    `tag`{.input}
    : An integer identifying the message

    `comm`{.input}
    : Communicator

    `error`{.output}
    : Error value; in C/C++ it’s the return value of the function, and
      in Fortran an additional output parameter

    `-`{.ghost}
    : `-`{.ghost}

# Receive operation {.split-definition}

MPI_Recv(`buffer`{.output}, `count`{.input}, `datatype`{.input}, `source`{.input}, `tag`{.input}, `comm`{.input}, `status`{.output})
  : `buffer`{.output}
    : Buffer for storing received data
    
    `count`{.input}
    : Number of elements in buffer, not the number of element that are
      actually received

    `datatype`{.input} 
    : Type of each element in buffer
    
    `source`{.input}
 	: Sender of the message
    
    `tag`{.input}
    : Number identifying the message

    `comm`{.input}
    : Communicator

    `status`{.output}
    : Information on the received message

    `error`{.output}
    : As for send operation

    `-`{.ghost}
    : `-`{.ghost}

# MPI datatypes

- MPI has a number of predefined datatypes to represent data
- Each C or Fortran datatype has a corresponding MPI datatype
    - C examples: `MPI_INT` for `int` and `MPI_DOUBLE` for
      `double`
    - Fortran example: `MPI_INTEGER` for `integer`
- One can also define custom datatypes

# More features in point-to-point communication {.section}

# Blocking routines & deadlocks

- MPI_Send and MPI_Recv are blocking routines
	- `MPI_Send` exits once the send buffer can be safely read and
      written to
	- `MPI_Recv` exits once it has received the message in the receive
      buffer
- Completion depends on other processes => risk for *deadlocks*
	- For example, all processes are in `MPI_Recv`
	- If deadlocked, the program is stuck forever

# Typical point-to-point communication patterns

![](images/comm_patt.svg){.center width=100%}

<br>

- Incorrect ordering of sends/receives may give a rise to a deadlock
  (or unnecessary idle time)

# Combined send & receive 

MPI_Sendrecv(`sendbuf`{.input}, `sendcount`{.input}, `sendtype`{.input}, `dest`{.input}, `sendtag`{.input}, `recvbuf`{.input}, `recvcount`{.input}, `recvtype`{.input}, `source`{.input}, `recvtag`{.input}, `comm`{.input}, `status`{.output})
  : `-`{.ghost}
    : `-`{.ghost}

- Sends one message and receives another one, with a single command
	- Reduces risk for deadlocks
- Parameters as in `MPI_Send` and `MPI_Recv`
- Destination rank and source rank can be same or different

# Special parameter values 

MPI_Send(`buffer`{.input}, `count`{.input}, `datatype`{.input}, `dest`{.input}, `tag`{.input}, `comm`{.input})
  : `-`{.ghost}
    : `-`{.ghost}

| Parameter          | Special value    | Implication                                  |
| ----------         | ---------------- | -------------------------------------------- |
| **`dest`{.input}** | `MPI_PROC_NULL`  | Null destination, no operation takes place   |

# Special parameter values

MPI_Recv(`buffer`{.output}, `count`{.input}, `datatype`{.input}, `source`{.input}, `tag`{.input}, `comm`{.input}, `status`{.output})
  : `-`{.ghost}
    : `-`{.ghost}

| Parameter             | Special value       | Implication                                  |
| ----------            | ----------------    | -------------------------------------------- |
| **`source`{.input}**  | `MPI_PROC_NULL`     | No sender=no operation takes place           |
|                       | `MPI_ANY_SOURCE`    | Receive from any sender                      |
| **`tag`{.input}**     | `MPI_ANY_TAG`       | Receive messages with any tag                |
| **`source`{.output}** | `MPI_STATUS_IGNORE` | Do not store any status data                 |

# Status parameter 

- The status parameter in `MPI_Recv` contains information about the
  received data after the call has completed, e.g.
	- Number of received elements
	- Tag of the received message
	- Rank of the sender
- In C the status parameter is a struct
- In Fortran it is an integer array of size `MPI_STATUS_SIZE`

# Status parameter

- Received elements  
&emsp;Use the function  
&emsp;**`MPI_Get_count`(`status`{.input}, `datatype`{.input}, `count`{.output})**
- Tag of the received message  
&emsp;C: `status.MPI_TAG`  
&emsp;Fortran: `status(MPI_TAG)`
- Rank of the sender  
&emsp;C: `status.MPI_SOURCE`  
&emsp;Fortran: `status(MPI_SOURCE)`

# Summary 

- Point-to-point communication = messages are sent between two MPI
  processes
- Point-to-point operations enable any parallel communication pattern
  (in principle)
    - `MPI_Send` and `MPI_Recv`
    - `MPI_Sendrecv`
- Employing special argument values may simplify the implementations
  of certain communication patterns
- Status parameter of `MPI_Recv` contains information about the
  message after the receive is completed

