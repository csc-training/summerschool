---
title:  User defined datatypes
author: CSC Summerschool
date:   2019-06
lang:   en
---

# User defined datatypes (part 1) {.section}

# MPI datatypes

- MPI datatypes are used for communication purposes
	- Datatype tells MPI where to take the data when sending or where
      to put data when receiving
- Elementary datatypes (`MPI_INT`, `MPI_REAL`, ...)
	- Different types in Fortran and C, correspond to languages basic
      types
	- Enable communication using contiguous memory sequence of
      identical elements (e.g. vector or matrix)

# Sending a matrix row (Fortran)

- Row of a matrix is not contiguous in memory in Fortran

<p>

![](images/layout.svg){.center width=50%}

<p>

- Several options for sending a row:
	- Use several send commands for each element of a row
	- Copy data to temporary buffer and send that with one send
      command
	- Create a matching datatype and send all data with one send
      command

# User-defined datatypes

- Use elementary datatypes as building blocks
- Enable communication of 
	- Non-contiguous data with a single MPI call, e.g. rows or columns
      of a matrix
	- Heterogeneous data (structs in C, types in Fortran)
	- Larger messages, count is `int` (32 bits) in C
- Provide higher level of programming
	- Code is more compact and maintainable
- Needed for getting the most out of MPI I/O

# User-defined datatypes

- User-defined datatypes can be used both in point-to-point
  communication and collective communication
- The datatype instructs where to take the data when sending or where
  to put data when receiving
	- Non-contiguous data in sending process can be received as
      contiguous or vice versa

# Using user-defined datatypes

- A new datatype is created from existing ones with a datatype constructor 	
	- Several routines for different special cases
- A new datatype must be committed before using it
    - **`MPI_Type_commit`(`newtype`{.input})**
- A type should be freed after it is no longer needed
    - **`MPI_Type_free`(`newtype`{.input})**

# Datatype constructors

| Datatype                   | Usage                                     |
|----------------------------|-------------------------------------------|
| `MPI_Type_contiguous`      | contiguous datatypes                      |
| `MPI_Type_vector`          | regularly spaced datatype                 |
| `MPI_Type_indexed`         | variably spaced datatype                  |
| `MPI_Type_create_subarray` | subarray within a multi-dimensional array |
| `MPI_Type_create_hvector`  | like vector, but uses bytes for spacings  |
| `MPI_Type_create_hindexed` | like index, but uses bytes for spacings   |
| `MPI_Type_create_struct`   | fully general datatype                    |

# MPI_TYPE_VECTOR

- Creates a new type from equally spaced identical blocks

<div class=column>
MPI_Type_vector(`count`{.input}, `blocklen`{.input}, `stride`{.input}, `oldtype`{.input}, `newtype`{.output})
  : `count`{.input}
    : number of blocks
  : `blocklen`{.input} 
    : number of elements in each block
  : `stride`{.input} 
    : displacement between the blocks
</div>
<div class=column>
<p>
![](images/type_vector.svg){.center width=100%}
</div>

# Example: sending rows of matrix in Fortran

```fortran
integer, parameter :: n=2, m=3
real, dimension(n,m) :: a
integer :: rowtype
! create a derived type
call mpi_type_vector(m, 1, n, mpi_real, rowtype, ierr)
call mpi_type_commit(rowtype, ierr)
! send a row
call mpi_send(a, 1, rowtype, dest, tag, comm, ierr)
! free the type after it is not needed
call mpi_type_free(rowtype, ierr)
```

<p>
![](images/layout.svg){.center width=50%}

# MPI_TYPE_INDEXED {.split-def-3}

- Creates a new type from blocks comprising identical elements
	- The size and displacements of the blocks may vary

MPI_Type_indexed(`count`{.input}, `blocklens`{.input}, `displs`{.input}, `oldtype`{.input}, `newtype`{.output})
  :	`count`{.input} 
    : number of blocks

    `blocklens`{.input}	
    : lengths of the blocks (array)

    `displs`{.input} 
    : displacements (array) in extent of oldtypes

    `oldtype`{.input}
    : original type
    
    `newtype`{.output}
    : new type

    `-`{.ghost}
    : `-`{.ghost}

<p>
![](images/type_indexed.svg){.center width=100%}

# Example: an upper triangular matrix

<div class="column">
```fortran
/* Upper triangular matrix */
double a[100][100];
int disp[100], blocklen[100], int i;
MPI_Datatype upper;
/* compute start and size of rows */
for (i=0; i<100; i++) {
    disp[i] = 100*i+i;
    blocklen[i] = 100-­i;
}
/* create a datatype for upper tr matrix */ 
MPI_Type_indexed(100,blocklen,disp,
    MPI_DOUBLE,&upper);
MPI_Type_commit(&upper);
/* ... send it ... */
MPI_Send(a,1,upper,dest, tag, MPI_COMM_WORLD);
MPI_Type_free(&upper);
```
</div>

<div class="column">
![](images/triangle.svg){.center width=65%}
</div>

# From non-contiguous to contiguous data

<div class=column>
![](images/contiguous.svg){.center width=100%}
</div>
<div class=column>
```c
if (myid == 0)
  MPI_Type_vector(n, 1, 2, 
                  MPI_FLOAT, &newtype)
  ...
  MPI_Send(A, 1, newtype, 1, ...)
else
  MPI_Recv(B, n, MPI_FLOAT, 0, ...)

```

```c
if (myid == 0)
  MPI_Send(A, n, MPI_FLOAT, 1, ...)
else
  MPI_Type_vector(n, 1, 2, MPI_FLOAT, 
                  &newtype)
  ...
  MPI_Recv(B, 1, newtype, 0, ...)

```
</div>

# Performance

- Main motivation for using datatypes is rarely performance – manual
  packing is often faster
- Performance depends on the datatype - more general datatypes are
  often slower
- Overhead is potentially reduced by:
	- Sending one long message instead of many small messages
	- Avoiding the need to pack data in temporary buffers
- Performance should be tested on target platforms

# Summary

- Derived types enable communication of non-contiguous or
  heterogeneous data with single MPI calls
	- Improves maintainability of program
	- Allows optimizations by the system 
	- Performance is implementation dependent
- Life cycle of derived type: create, commit, free
- MPI provides constructors for several specific types
