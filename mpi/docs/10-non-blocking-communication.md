---
title:  Further communication modes
event:  CSC Summer School in High-Performance Computing 2022
lang:   en
---

# Non-blocking communication {.section}

# Non-blocking communication

- Non-blocking communication operations return immediately and perform sending/receiving in the background
    - `MPI_Isend` & `MPI_Irecv`
- Enables some computing concurrently with communication
- Avoids many common dead-lock situations
- Collective operations are also available as non-blocking versions

# Non-blocking send

Parameters similar to **`MPI_Send`** but has an additional request parameter.

<p>
MPI_Isend(`buffer`{.input}, `count`{.input}, `datatype`{.input}, `dest`{.input}, `tag`{.input}, `comm`{.input}, `request`{.output})
  : `buffer`{.input}
    : send buffer that must not be written to until one has checked
      that the operation is over
  : `request`{.output}
    : a handle that is used when checking if the operation has
      finished (`type(mpi_request)` in Fortran, `MPI_Request` in C)

# Non-blocking receive

Parameters similar to **`MPI_Recv`** but has no status parameter.

<p>
MPI_Irecv(`buffer`{.output}, `count`{.input}, `datatype`{.input}, `source`{.input}, `tag`{.input}, `comm`{.input}, `request`{.output})
  : `buffer`{.output}
    : receive buffer guaranteed to contain the data only after one has
      checked that the operation is over
  : `request`{.output}
    : a handle that is used when checking if the operation has
      finished

# Non-blocking communication

- Important: Send/receive operations have to be finalized
    - `MPI_Wait`, `MPI_Waitall`, ...
        - Waits for the communication started with `MPI_Isend` or
          `MPI_Irecv` to finish (blocking)
    - `MPI_Test`, ...
        - Tests if the communication has finished (non-blocking)
    - Remember: successfully finished send does not mean successful receive!
- You can mix non-blocking and blocking routines
    - e.g., receive a message sent by `MPI_Isend` with `MPI_Recv`

# Wait for non-blocking operation

MPI_Wait(`request`{.input}, `status`{.output})
  : `request`{.input}
    : handle of the non-blocking communication
  : `status`{.output}
    : status of the completed communication, see `MPI_Recv`

A call to `MPI_Wait` returns when the operation identified by request is complete

# Wait for non-blocking operations

MPI_Waitall(`count`{.input}, `requests`{.input}, `status`{.output})
  : `count`{.input}
    : number of requests
  : `requests`{.input}
    : array of requests
  : `status`{.output}
    : array of statuses for the operations that are waited for

A call to `MPI_Waitall` returns when all operations identified by the array of requests are complete.

# Non-blocking test for non-blocking operations

MPI_Test(`request`{.input}, `flag`{.output}, `status`{.output})
  : `request`{.input}
    : request
  : `flag`{.output}
    : True if the operation has completed
  : `status`{.output}
    : status for the completed operation

A call to `MPI_Test` is non-blocking. It allows one to schedule alternative activities while periodically checking for completion.

`MPI_Probe` is a similar kind of operation (see later slides).

# Typical usage pattern

<div class=column>
**`MPI_Irecv`(<font color="green">ghost_data</font>)**
**`MPI_Isend`(<font color="purple">border_data</font>)**
**`compute`(<font color="DeepSkyBlue">ghost_independent_data</font>)**
**`MPI_Waitall`**
**`compute`(<font color="purple">border_data</font>)**
</div>
<div class=column>
![](img/usage_pattern.png){.center width=100%}
</div>

# Additional completion operations

| Routine      | Meaning                                            |
|--------------|----------------------------------------------------|
| MPI_Waitany  | Waits until any one operation has completed        |
| MPI_Waitsome | Waits until at least one operation has completed   |
| MPI_Test     | Tests if an operation has completed (non-blocking) |
| MPI_Testall  | Tests whether a list of operations have completed  |
| MPI_Testany  | Like Waitany but non-blocking                      |
| MPI_Testsome | Like Waitsome but non-blocking                     |
| MPI_Probe    | Check for incoming messages without receiving them |

# Wait for non-blocking operations {.split-definition}

**`MPI_Waitany`(`count`{.input}, `requests`{.input}, `index`{.output}, `status`{.output})**
  : `count`{.input}
    : number of requests

    `requests`{.input}
    : array of requests

    `index`{.output}
    : index of request that completed

    `status`{.output}
    : status for the completed operations

<p>
A call to `MPI_Waitany` returns when one operation identified by the array of requests is complete.

# Wait for non-blocking operations {.split-definition}

MPI_Waitsome(`count`{.input}, `requests`{.input}, `done`{.output}, `index`{.output}, `status`{.output})
  : `count`{.input}
    : number of requests

    `requests`{.input}
    : array of requests

    `done`{.output}
    : number of completed requests

    `index`{.output}
    : array of indexes of completed requests

    `status`{.output}
    : array of statuses of completed requests

    `-`{.ghost}
    : `-`{.ghost}

<p>
Returns when one or more operations is/are complete.

# Message Probing {.split-definition}

MPI_Iprobe(`source`{.input}, `tag`{.input}, `comm`{.input}, `flag`{.output}, `status`{.output})
  : `source`{.input}
    : rank of sender (or `MPI_ANY_SOURCE`)

    `tag`{.input}
    : message of the tag (or `MPI_ANY_TAG`)

    `comm`{.input}
    : communicator

    `flag`{.output}
    : true if there is a message that matches the pattern and can be
      received

    `status`{.output}
    : status object

    `-`{.ghost}
    : `-`{.ghost}

<p>
Allows incoming messages to be checked, without actually receiving them.

# Non-blocking collectives

- Non-blocking collectives (“``I``-collectives”) enable the overlapping of communication and computation together with the benefits of collective communication.

- Same syntax as for blocking collectives, besides
    - “``I``” at the front of the name (`MPI_Alltoall` -> `MPI_Ialltoall`)
    - Request parameter at the end of the list of arguments
    - Completion needs to be waited

# Non-blocking collectives

- Restrictions
    - Have to be called in same order by all ranks in a communicator
    - Mixing of blocking and non-blocking collectives is not allowed

# Non-blocking collectives

![](img/non_blocking_large.png){.center width=100%}

![](img/blue_arrow.png){width=1%} (Computation) work 1
![](img/green_arrow.png){width=1%} (Computation) work 2, not
involving data in the ``Allreduce`` operation

# Example: Non-blocking broadcasting {.split-definition}

MPI_Ibcast(`buf`{.input}`fer`{.output}, `count`{.input}, `datatype`{.input}, `root`{.input}, `comm`{.input}, `request`{.output})
  : `buf`{.input}`fer`{.output}
    : data to be distributed

    `count`{.input}
    : number of entries in buffer

    `datatype`{.input}
    : data type of buffer

    `root`{.input}
    : rank of broadcast root

    `comm`{.input}
    : communicator

    `request`{.output}
    : a handle that is used when checking if the operation has finished


# Persistent communication {.section}

# Persistent communication

- Often a communication with same argument list is repeatedly executed
- It may be possible to optimize such pattern by persistent communication requests
    - Can be thought as a ”communication port”
- Usage:
    - Create requests: `MPI_Send_init` & `MPI_Recv_init`
        - Initiation and starting of communication separated in addition to communication and completion
    - Start communication: `MPI_Start` / `MPI_Startall`
    - Complete communication: `MPI_Wait` / `MPI_Waitall`
- Recently published (June 2021) MPI 4.0 includes also persistent
  collectives
    - Not supported by all implementations yet

# Persistent point-to-point communication

```c
MPI_Request recv_req, send_req;
...
// Initialize send/request objects
MPI_Recv_init(buf1, cnt, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, &recv_req);
MPI_Send_init(buf2, cnt, MPI_DOUBLE, dst, tag, MPI_COMM_WORLD, &send_req);
for (int i=1; i<BIGNUM; i++){
// Start communication described by recv_obj and send_obj
    MPI_Start(&recv_req);
    MPI_Start(&send_req);
    // Do work, e.g. update the interior domains
    ...
    // Wait for send and receive to complete
    MPI_Wait(&send_req, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
}
//Clean up the requests
MPI_Request_free (&recv_req); MPI_Request_free (&send_req);
```

# Neighborhood collectives {.section}

# Neighborhood collectives

- Neighborhood collectives build on top of process topologies
- Provide optimization possibilities for MPI library for communication
  patterns involving neighbors
    - Nearest neighbors in cartesian topology
        - Processes connected by a edge in a general graph
- Similar to ordinary collectives, all tasks within a communicator
  need to call the routine
- Possible to have multidimensional halo-exchange with a single MPI call

# Neighborhood collectives in cartesian grid

<div class=column>
- Only nearest neighbors, *i.e.* those corresponding to
  `MPI_Cart_shift` with displacement=1.
- Boundaries in finite dimensions treated as like with `MPI_PROC_NULL`
</div>

<div class=column>
![](img/cartesian-neighborhood.png){width=90%}
</div>

# Neighborhood collectives

- Two main neighborhood operations
    - `MPI_Neighbor_allgather` : send same data to all neighbors, receive different
      data from neighbors
        - `MPI_Neighbor_alltoall` : send and receive different data
      between all the neighbors
- Also variants where different number or type of elements is
  communicated
- Non-blocking versions with similar semantics than non-blocking
  collectives
    - Request parameter at the end of the list of arguments

# Summary

- Non-blocking communication is often useful way to do point-to-point
  communication in MPI.
- Non-blocking communication core features
    - Open receives with `MPI_Irecv`
    - Start sending with `MPI_Isend`
    - Possibly do something else while the communication takes place
    - Complete the communication with `MPI_Wait` or a variant
- Collective operations can also be done in non-blocking mode

# Summary

- In persistent communication the communication pattern remains constant
- All the parameters for the communication are set up in the initialization phase
    - Communication is started and finalized in separate steps
- Neighborhood collectives enable communication between neighbors in process topology
  with a single MPI call
- Persistent and neighborhood communication provide optimization opportunities for MPI
  library

# Further MPI topics

- One-sided communication
- MPI shared memory programming
- MPI error handling
- Intercommunicators
- Point-to-point communication modes
- Dynamic process creation
