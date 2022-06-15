---
title:  OpenMP reductions and execution control
event:  CSC Summer School in High-Performance Computing 2022
lang:   en
---

# OpenMP reductions {.section}

# Race condition in reduction

- Race conditions take place when multiple threads read and write a variable
  simultaneously, for example:

```fortran
asum = 0.0d0
!$omp parallel do shared(x,y,n,asum) private(i)
  do i = 1, n
     asum = asum + x(i)*y(i)
  end do
!$omp end parallel do
```

- Random results depending on the order the threads access **asum**
- We need some mechanism to control the access


# Reductions

- Summing elements of array is an example of reduction operation

$$
S = \sum_{j=1}^{N} A_j = \sum_{j=1}^{\frac{N}{2}} A_j +
\sum_{j=\frac{N}{2} + 1}^{N} A_j = B_1 + B_2 = \sum_{j=1}^{2} B_j
$$

- OpenMP provides support for common reductions within parallel regions and
  loops with the reduction clause


# Reduction clause

`reduction(operator:list)`
  : Performs reduction on the (scalar) variables in list
  : `-`{.ghost}

- Private reduction variable is created for each thread's partial result
- Private reduction variable is initialized to operator's initial value
- After parallel region the reduction operation is applied to private
  variables and result is aggregated to the shared variable


# Reduction operators in C/C++

<div class="column">

| Operator | Initial value |
|----------|---------------|
| `+`     | `0`           |
| `-`      | `0`           |
| `*`      | `1`           |
| `&&`     | `1`           |
| `||`     | `0`           |

</div>
<div class="column">

| Bitwise Operator | Initial value |
|----------|---------------|
| `&`      | `~0`          |
| `|`      | `0`           |
| `^`      | `0`           |

</div>


# Reduction operators in Fortran

<small>
<div class="column">

| Operator         | Initial value |
|------------------|---------------|
| `+`              | `0`           |
| `-`              | `0`           |
| `*`              | `1`           |
| `max`            | least         |
| `min`            | largest       |
| `.and.`          | `.true.`      |
| `.or.`           | `.false.`     |
| `.eqv.`          | `.true.`      |
| `.neqv.`         | `.false.`     |

</div>
<div class="column">

| Bitwise Operator | Initial value |
|------------------|---------------|
| `.iand.`           | all bits on   |
| `.ior.`            | `0`           |
| `.ieor.`           | `0`           |

</div>

</small>


# Race condition avoided with reduction clause

```fortran
!$omp parallel do shared(x,y,n) private(i) reduction(+:asum)
  do i = 1, n
     asum = asum + x(i)*y(i)
  end do
!$omp end parallel do
```
```c
#pragma omp parallel for shared(x,y,n) private(i) reduction(+:asum)
for(i=0; i < n; i++) {
  asum = asum + x[i] * y[i];
}
```

# OpenMP execution controls {.section}

# Execution controls

- Sometimes a part of a parallel region should be executed only by the
  master thread or by a single thread at a time
    - IO, initializations, updating global values, etc.
    - Remember the synchronization!
- OpenMP provides clauses for controlling the execution of code blocks


# Execution control constructs

`barrier`
  : `-`{.ghost}

- When a thread reaches a barrier it only continues after all the threads in
  the same thread team have reached it
    - Each barrier must be encountered by all threads in a team, or none at
      all
    - The sequence of work-sharing regions and barrier regions encountered
      must be same for all threads in a team
- Implicit barrier at the end of: `do`, `parallel`, `single`, `workshare`
  unless a `nowait` clause is specified


# Execution control constructs

<div class=column>
`master`
  : `-`{.ghost}

- Specifies a region that should be executed only by the master thread

- Other threads do not wait, *i.e.* no implicit barrier at the end

- Deprecated in OpenMP 5.1 and replaced with `masked`
</div>
<div class=column>
`single`
  : `-`{.ghost}

- Specifies that a region should be executed only by a single (arbitrary)
  thread

- Other threads wait (implicit barrier) unless a `nowait` clause is specified
</div>

# Summary

- Several parallel reduction operators available via `reduction` clause
- OpenMP has many synchronization pragmas
    - Barrier
    - Single and Master




# Web resources

- OpenMP homepage: <http://openmp.org/>
- Online tutorials: <https://www.openmp.org/resources/tutorials-articles/>
