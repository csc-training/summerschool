---
title:  "Extra: Persistent communication"
event:  CSC Summer School in High-Performance Computing 2024
lang:   en
---

# Persistent communication

- Often a communication with same argument list is repeatedly executed
- It may be possible to optimize such pattern by persistent communication requests
    - Can be thought as a ”communication port”
- Three separate phases:
    1. Initiation of communication
    2. Starting of communication
    3. Completing communication
- Recently published MPI 4.0 includes also persistent collectives
    - Not supported by all implementations yet

# Persistent communication

- Initiate communication by creating requests
    - `MPI_Send_init` and `MPI_Recv_init`
    - Same arguments as in `MPI_Isend` and `MPI_Irecv`
- Start communication
    - `MPI_Start` / `MPI_Startall`
    - Request or array of requests as argument
- Complete communication
    - `MPI_Wait` / `MPI_Waitall`
    - Same as in standard non-blocking communication

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

# Summary

- In persistent communication the communication pattern remains constant
- All the parameters for the communication are set up in the initialization phase
    - Communication is started and finalized in separate steps
- Persistent communication provides optimization opportunities for MPI library
