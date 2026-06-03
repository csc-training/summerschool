---
# SPDX-FileCopyrightText: 2019 CSC - IT Center for Science Ltd. <www.csc.fi>
#
# SPDX-License-Identifier: CC-BY-4.0

title: Application performance
event: CSC Summer School in High-Performance Computing 2026
lang:  en
---

# Performance analysis ("profiling")

My code is too slow! What do I do?

- Find out *where* the slow code segments are
- Figure out *why* it's slow
    - Suboptimal algorithm?
    - Inefficient use of hardware features?
- Focus optimization efforts where it matters!
    - Long function that is only called once VS a short function that is called 1 000 000 times?

Let's look at common **profiling methods** and **tools** for systematically answering these questions.


# Collecting data by tracing

- Place markers or timestamps in code, either manually or automatically with tracing tools
- Measure the wall-clock difference between two markers
- Caution: tracing *everything* is slow and can produce enormous output files

![](img/trace_example.png)
<small>*Example traces from two MPI tasks*</small>


# Collecting data by sampling

- Periodically interrupt program execution and record where it is
- Gives a *statistical* performance profile of a program: hot code paths emerge naturally

<small>
```
    9216685 function calls (9214011 primitive calls) in 83.535 seconds

   Ordered by: internal time
   List reduced from 1639 to 6 due to restriction <6>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        8   19.076    2.384   19.076    2.385 _decomp.py:284(eigh)
        6   14.773    2.462   14.773    2.462 {method 'construct_density' of 'LocalizedFunctionsCollection' objects}
        4   11.353    2.838   11.353    2.838 {method 'calculate_potential_matrices' of 'LocalizedFunctionsCollection' objects}
        8   10.885    1.361   11.644    1.456 hamiltonian.py:57(_calculate_matrix_without_kinetic)
       30    3.553    0.118    3.553    0.118 {built-in method _gpaw.mmm}
    72005    3.043    0.000    3.043    0.000 {method 'calculate' of 'XCFunctional' objects}
```
</small>
*Sampling results from the Python cProfile profiler*


# Hardware counters

Tracking hardware events give hints about *why* a particular code segment takes as long as it does.

- CPU cycles
- Cache misses
- Memory page faults
- Branch mispredictions

# Profiling tools

TODO list common profiling tools and their main differences/use cases


# Demo: tracing with Score-P and Vampir

Profiling the message-chain MPI exercise
