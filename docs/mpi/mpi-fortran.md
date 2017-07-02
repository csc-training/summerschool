# MPI Cheatsheet for Fortran
A short collection of different MPI functions and their respective inputs for `Fortran`.

## Interfaces for basic MPI operations
```fortran
integer :: comm, Nrank, rank, rc

call mpi_init(rc) 
call mpi_comm_size(comm, Nrank, rc) 
call mpi_comm_rank(comm, rank, rc) 
call mpi_barrier(comm, rc) 
call mpi_finalize(rc)
```
 
## Datatypes

MPI Type             | Fortran Type
--------             | ----------
MPI_CHARACTER        | `character`
MPI_INTEGER          | `integer`
MPI_REAL             | `real`
MPI_REAL8            | `real*8`
MPI_DOUBLE_PRECISION | `double precision`
MPI_COMPLEX          | `complex`
MPI_DOUBLE_COMPLEX   | `double complex`
MPI_LOGICAL          | `logical`
MPI_BYTE             |

## Point-to-Point communication

##### `send` and `receive`:
```fortran
integer :: count, datatype, source, dest, tag, comm, rc
integer :: status(MPI_STATUS_SIZE)
type :: sendbuf(*), recvbuf(*)

call mpi_send(sendbuf, count, datatype, dest, tag, comm, rc) 
call mpi_recv(recvbuf, count, datatype, source, tag, comm, status, rc) 
```

##### Combined `send` and `receive`:
```fortran
integer :: sendcount, sendtype, dest, sendtag, recvcount, recvtype, source, recvtag, comm, rc
integer :: status(MPI_STATUS_SIZE)
type :: sendbuf(*), recvbuf(*)

call mpi_sendrecv(
    sendbuf, 
    sendcount, 
    sendtype, 
    dest, 
    sendtag, 
    recvbuf, 
    recvcount, 
    recvtype, 
    source, 
    recvtag, 
    comm, 
    status, 
    rc
    )
```

