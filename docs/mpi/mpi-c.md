# MPI Cheatsheet for C
A short collection of different MPI functions and their respective inputs for `C`.

## Interfaces for basic MPI operations
```c
int MPI_Init(int *argc, char **argv)
int MPI_Comm_size(MPI_Comm comm, int *size)
int MPI_Comm_rank(MPI_Comm comm, int *rank)
int MPI_Barrier(MPI_Comm comm)
int MPI_Finalize()
```
 
## Datatypes

MPI Type           | C Type
--------           | ----------
MPI_CHAR           | `signed char`
MPI_SHORT          | `short int`
MPI_INT            | `int`
MPI_LONG           | `long int`
MPI_UNSIGNED_SHORT | `unsigned short int`
MPI_UNSIGNED_INT   | `unsigned int`
MPI_UNSIGNED_LONG  | `unsigned long int`
MPI_FLOAT          | `float`
MPI_DOUBLE         | `double`
MPI_LONG_DOUBLE    | `long double`
MPI_BYTE           |

## Point-to-Point communication

##### `send` and `receive`:
```c
int MPI_Send(
    void *buffer,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm
    )
```

```c
int MPI_Recv( 
    void *buf, 
    int count, 
    MPI_Datatype datatype, 
    int source, 
    int tag, 
    MPI_Comm comm, 
    MPI_Status *status
    )
```

##### Combined `send` and `receive`:
```c
int MPI_Sendrecv(
    void *sendbuf, 
    int sendcount, 
    MPI_Datatype sendtype, 
    int dest, 
    int sendtag, 
    void *recvbuf, 
    int recvcount, 
    MPI_Datatype,
    recvtype, 
    int source, 
    int recvtag, 
    MPI_Comm comm, 
    MPI_Status *status
    )
```

```c
int MPI_Get_count(
    MPI_Status *status, 
    MPI_Datatype datatype, 
    int *count
    )
```
    
    
    
    
    
    
    
    
 



