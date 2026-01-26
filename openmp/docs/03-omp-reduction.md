<!--
SPDX-FileCopyrightText: 2010 CSC - IT Center for Science Ltd. <www.csc.fi>

SPDX-License-Identifier: CC-BY-4.0
-->

---
title:  OpenMP reduction
event:  CSC Summer School in High-Performance Computing 2026
lang:   en
---


# OpenMP reductions {.section}

# Race condition in reduction

- Race conditions take place when multiple threads read and write a variable
  simultaneously, for example:

```fortranfree
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

- Summing the elements of an array is an example of a reduction operation
  $$
  S = \sum_{j=1}^{N} A_j = \sum_{j=1}^{\frac{N}{2}} A_j +
  \sum_{j=\frac{N}{2} + 1}^{N} A_j = B_1 + B_2 = \sum_{j=1}^{2} B_j
  $$

- OpenMP provides support for common reductions within parallel regions and
  loops


# Reduction clause

`reduction(operator:list)`
  : Performs reduction on the (scalar) variables in list

<br>

- Private reduction variable is created for each thread's partial result
- Private reduction variable is initialized to operator's initial value
- After parallel region the reduction operation is applied to private
  variables and result is aggregated to the shared variable


# Reduction operators in C/C++

<div class="column">

| Operator | Initial value |
|----------|---------------|
| `+`      | `0`           |
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

<div class="column" style="font-size:0.8em">

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

<div class="column" style="font-size:0.8em">

| Bitwise Operator | Initial value |
|------------------|---------------|
| `.iand.`           | all bits on   |
| `.ior.`            | `0`           |
| `.ieor.`           | `0`           |

</div>


# Race condition avoided with reduction clause

```fortranfree
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

# Summary

- Several parallel reduction operators available via `reduction` clause
