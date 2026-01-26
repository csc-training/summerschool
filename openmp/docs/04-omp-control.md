<!--
SPDX-FileCopyrightText: 2010 CSC - IT Center for Science Ltd. <www.csc.fi>

SPDX-License-Identifier: CC-BY-4.0
-->

---
title:  OpenMP execution control
event:  CSC Summer School in High-Performance Computing 2026
lang:   en
---


# OpenMP execution control {.section}

# How to distribute work?

- Each thread executes the same code within the parallel region
- OpenMP provides several constructs for controlling work distribution
    - `for`/`do` construct
    - `workshare` construct (Fortran only)
    - `single`/`master`/`masked` construct
    - `task` construct
    - `distribute` construct (for GPUs)
    - `loop` construct
    - `sections` construct
- Thread ID can be queried and used for distributing work manually
  (similar to MPI rank)


# Controlling number of teams and threads

- By default, the number of threads is up to the implementation to decide
- The `num_threads` clause for `parallel` construct can be used to specify number of threads

<div class=column>
```c++
#pragma omp parallel num_threads(128)
{
  ...
}
```
</div>

<div class=column>
```fortranfree
!$omp parallel num_threads(128)
  ...

!$omp end parallel
```
</div>


# Execution controls

- Sometimes a part of a parallel region should be executed only by the
  master thread or by a single thread at a time
    - IO, initializations, updating global values, etc.
    - remember to synchronize!
- OpenMP provides clauses for controlling the execution of code blocks


# Execution control constructs

`barrier`
  : `-`{.ghost}

- When a thread reaches a barrier it only continues after all the threads in
  the same thread team have reached it
    - each barrier must be encountered by all threads in a team, or none at
      all
    - the sequence of work-sharing regions and barrier regions encountered
      must be same for all threads in a team
- Implicit barrier at the end of: `for`/`do`, `parallel`, `single`, `workshare`
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


# Execution control constructs

<div class=column>
`critical`
  : `-`{.ghost}

- A section that is executed by only one thread at a time
- No implicit barrier at the end
</div>

<div class=column>
`atomic`
  : `-`{.ghost}

- Strictly limited construct to update a single value, can not be applied to
  code blocks
- Only guarantees atomic update, does not protect function calls
- Can be faster on hardware platforms that support atomic updates
</div>


# Summary

- OpenMP has many synchronization constructs
    - `barrier`
    - `single` and `master/masked`
    - `critical` and `atomic`

