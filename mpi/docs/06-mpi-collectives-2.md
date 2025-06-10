---
title:  Collective communication II
event:  CSC Summer School in High-Performance Computing 2025
lang:   en
---

# Outline

- Collective reduction routines
- Non-blocking collective routines in MPI

# Collective reductions {.section}

# What is reduction?

- Reduction in functional programming: Reduce a set of numbers into a smaller set of numbers using some function
    - Example: `sum([1, 2, 3]) = 6` reduces three numbers to one by adding them

- In parallel programming, (collective) **reduction** refers to a computational operation performed on data distributed over many processes
    - Example: Get value of some variable `a` on all processes and add them together

- MPI provides a collective `MPI_Reduce` function that supports common reduction operations

# Reduction illustration

- Applies an operation to data scattered over processes and places the result in a single process

![](img/reduce.png){.center width=80%}

# Reduction in MPI

MPI_Reduce(`sendbuf`{.input}, `recvbuf`{.output}, `count`{.input}, `datatype`{.input}, `op`{.input}, `root`{.input}, `comm`{.input})
  : Combines values to the root process from all processes of the group

<p>
- `op` is an *operation code* specifying the reduction operation (sum, product, ...)
- Only the root process gets the reduced result
- Prefer `MPI_Reduce` instead of manually gathering and processing data on a single process
- Demo: `reduce.c`


# Available reduction operations

<div class=column>
| Operation    | Meaning              |
|--------------|----------------------|
| `MPI_MAX`    | Max value            |
| `MPI_MIN`    | Min value            |
| `MPI_SUM`    | Sum                  |
| `MPI_PROD`   | Product              |
| `MPI_MAXLOC` | Max value + location |
| `MPI_MINLOC` | Min value + location |
</div>
<div class=column>
| Operation  | Meaning      |
|------------|--------------|
| `MPI_LAND` | Logical AND  |
| `MPI_BAND` | Bitwise AND  |
| `MPI_LOR`  | Logical OR   |
| `MPI_BOR`  | Bitwise OR   |
| `MPI_LXOR` | Logical XOR  |
| `MPI_BXOR` | Bitwise XOR  |
</div>


# Global reduction

MPI_Allreduce(`sendbuf`{.input}, `recvbuf`{.output}, `count`{.input}, `datatype`{.input}, `op`{.input}, `comm`{.input})
  :  Combines values from all processes and distributes the result back to all processes

<p>
- All processes get the reduced result
- Similar to `MPI_Reduce` + `MPI_Bcast` but more efficient

<p>
- Demo: `reduce.c`


# Allreduce example: parallel dot product

<div class=column>
```fortranfree
real :: a(1024), aloc(128)
...
if (rank==0) then
    call random_number(a)
end if
call mpi_scatter(a, 128, MPI_INTEGER, &
                 aloc, 128, MPI_INTEGER, &
                 0, MPI_COMM_WORLD, rc)
rloc = dot_product(aloc, aloc)
call mpi_allreduce(rloc, r, 1, MPI_REAL, &
                   MPI_SUM, MPI_COMM_WORLD, &
                   rc)
```
</div>
<div class=column>
```
> srun -n 8 ./mpi_pdot
 id= 6 local= 39.68326  global= 338.8004
 id= 7 local= 39.34439  global= 338.8004
 id= 1 local= 42.86630  global= 338.8004
 id= 3 local= 44.16300  global= 338.8004
 id= 5 local= 39.76367  global= 338.8004
 id= 0 local= 42.85532  global= 338.8004
 id= 2 local= 40.67361  global= 338.8004
 id= 4 local= 49.45086  global= 338.8004
```
</div>


# In-place collective operations

- What if we use the input buffer as the output buffer?<br>
`call mpi_allreduce(a, a, n, mpi_real,...`

- Special clause for this purpose: `MPI_IN_PLACE`
    - Replacing the send buffer with `MPI_IN_PLACE` informs MPI that the buffers are the same
- Demo: `reduce.c`

# Non-blocking collectives {.section}

# Non-blocking collectives

- Non-blocking collectives (“``I``-collectives”) enable the overlapping of communication and computation together with the benefits of collective communication.

- Same syntax as for blocking collectives, besides
    - “``I``” at the front of the name (`MPI_Alltoall` -> `MPI_Ialltoall`)
    - Request parameter at the end of the list of arguments
    - Completion needs to be waited
    - See earlier slides on non-blocking communication

# Example: Non-blocking broadcasting

MPI_Ibcast(`buf`{.input}`fer`{.output}, `count`{.input}, `datatype`{.input}, `root`{.input}, `comm`{.input}, `request`{.output})
: Broadcasts data from the root process to all other processes of the group

<p>
- Request parameter at the end of the list of arguments in comparison to `MPI_Bcast`


# Restrictions on non-blocking collectives

- Have to be called in same order by all ranks in a communicator
- Mixing of blocking and non-blocking collectives is **not** allowed


# Illustration: blocking vs non-blocking reduction

![](img/non_blocking_large.png){.center width=100%}

![](img/blue_arrow.png){width=1%} (Computation) work 1
<br>
![](img/green_arrow.png){width=1%} (Computation) work 2, not
involving data in the `MPI_Allreduce` operation

# Non-blocking collectives

- Demo: `nonblocking_collectives.c`


# Summary {.section}

# Summary: reductions and non-blocking collectives

- Collective reductions allow efficient collective computation
- The `MPI_IN_PLACE` clause can be useful when you want to overwrite your send buffer in certain collective operations
- Collective operations can also be done in non-blocking mode
