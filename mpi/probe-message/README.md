## Probe message

This exercise demonstrates the use of `MPI_Probe()` together with the `MPI_Status` object
to extract information about an incoming message. Have a look at the example program:
    - C++: [probe.cpp](probe.cpp)
    - Fortran: [probe.F90](probe.F90)

This program is intended to be ran with (at least) 2 MPI processes. Process with rank 1
generates a message of random length and sends it to rank 0. Rank 0 wants to receive this message,
however it does *not know much memory it needs to allocate for the message*.

If you compile and run the example program, you should see the `MPI_Recv` fail
with an error message like `*** MPI_ERR_TRUNCATE: message truncated`,
because the receive buffer was too small to hold the full message. Your task is to fix this:
- Use `MPI_Probe` to probe for the incoming message without receiving it,
and store information about the message in an `MPI_Status` variable.
- Use `MPI_Get_count` to find the incoming message size.
- Reallocate the receive buffer accordingly and receive the message.


### Bonus technical information

The (MPI specification)[https://www.mpi-forum.org/] for `MPI_Recv` states:

>"The receive buffer consists of the storage containing count consecutive elements of the
type specified by datatype, starting at address buf. The length of the received message must
be less than or equal to the length of the receive buffer. An overflow error occurs if all
incoming data does not fit, without truncation, into the receive buffer.
If a message that is shorter than the receive buffer arrives, then only those locations
corresponding to the (shorter) message are modified."

This means that the programmer is responsible for allocating a receive buffer that is *at least*
as large as the incoming message. Receiving just a small part of an incoming message is not possible with `MPI_Recv`.
Both `OpenMPI` and `MPICH` implementations fail with `MPI_ERR_TRUNCATE` if the buffer is too small for the full message.

However, it is completely allowed to use a receive buffer that is "too large". This can be useful in some situations.
Eg: your program may allocate a large buffer and use it to receive messages of varying sizes,
without having to constantly reallocate the buffer.
