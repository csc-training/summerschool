---
title:  Introduction to GPUs in HPC
author: CSC Summerschool
date:   2019-07
lang:   en
---


# High-performance computing

- A special branch of scientific computing -- high-performance
  computing (HPC) or *supercomputing* -- that refers to computing with
  supercomputer systems, is the scientific instrument of the future
- It offers a promise of breakthroughs in many major challenges that
  humankind faces today
- Useful through various disciplines
- Compute resources based on a very high level of parallelism to reach
  high performance


# HPC through the ages

FIXME: missing figure


# Accelerators

- Specialized parallel hardware for floating point operations
    - Co-processors for traditional CPUs
    - Based on highly parallel architectures
    - Graphics processing units (GPU) have been the most common
      accelerators during the last few years
- Promises
    - Very high performance per node
- Usually major rewrites of programs required


# Accelerators

FIXME: missing figure


# HPC through the ages

FIXME: missing figure


# Accelerator model today

- Add-on PCIe cards
- Local memory
    - Smaller than main memory
    - Very high bandwidth (up to 900 GB/s on latest generation),
    - Latency high compared to compute performance
- Data must be copied over the PCIe bus (or NVLink on some platforms)

FIXME: missing figure


# GPU architecture -- Kepler Case study

- Designed for running tens of thousands of threads simultaneously on
  thousands of cores
- Very small penalty for switching threads
- Running large amounts of threads hides memory access penalties
- Very expensive to synchronize all threads


# GPU architecture -- Kepler Case study

- 15 symmetric multi processor units (SMX), each comprising many smaller
  Cuda cores
    - 3072 single precision cores
    - 1024 double precision cores
- Common L2 cache for all multi processors
- 6 memory controllers for GDDR5

FIXME: update and add missing figures


# GPU architecture -- Kepler Case study - SMX

- 192 single precision cores
- 64 double precision cores
- L1 cache that can also be used in a manual mode (shared memory)
- All execution is done in terms of 32 threads -- a warp

FIXME: update and add missing figures


# GPU architecture -- Kepler Case study - warp

- In a warp 32 threads compute the same instruction on different data
  (SIMT)
    - In case of divergence (if...) computation serializes
    - Best performance if memory access is contiguous in memory, and not
      scattered (coalesced )
- Up to 4 warps can execute simultaneously

FIXME: update and add missing figures


# Nvidia Pascal architecture

- Faster performance
    - Double precision ~ 5 TFlops
    - Half-precision -- 4x faster than double precision.
- Faster memory
    - HBM -- Stacked memory providing very high BW (~ 900 GB/s)
- Faster interconnects
    - Nvlink, new intra-GPU links
- And other improvements
    - Page faults, etc...

FIXME: update and add missing figures


# Challenges in using Accelerators

Applicability
  : Is your algorithm suitable for GPU?
Programmability
  : Is the programming effort acceptable?
Portability
  : Rapidly evolving ecosystem and incompatibilities between vendors.
Availability
  : Can you access a (large scale) system with GPUs?
Scalability
  : Can you scale the GPU software efficiently to several nodes?


# Using GPUs

1. Use existing GPU applications
2. Use accelerated libraries
3. Directive based methods
    - OpenMP, **OpenACC**
4. Use lower level language
    - CUDA, OpenCL

FIXME: add missing figure


# Directive-based accelerator languages

- Annotating code to pinpoint accelerator-offloadable regions
- OpenACC standard created in Nov 2011
    - Focus on optimizing productivity (reasonably good performance with
      minimal effort)
    - Current standard is 2.6 (November 2017)
    - Also support for CPU, AMD GPUs, Xeon Phis
- OpenMP
   - Earlier only threading for CPUs
   - 4.5 also includes for the first time some support for accelerators
   - Dominant directive approach in the future?


# GPUs at CSC

- At CSC there are ~150 GPUs
- 12 nodes with
    - 2 x K80, in total 4 GPUs each
    - 2 x Haswell CPU, 24 cores in total
- 26 nodes with
    - 4 x P100

FIXME: update!


# Parallel computing concepts {.section}


# Computing in parallel

- Serial computing
    - Single processing unit ("core") is used for solving a problem

FIXME: add missing figure


# Computing in parallel

- Parallel computing
    - A problem is split into smaller subtasks
    - Multiple subtasks are processed *simultaneously* using multiple
      cores

FIXME: add missing figure


# Exposing parallelism

- Data parallelism
    - Data is distributed to processor cores
    - Each core performs simultaneously (nearly) identical operations with
      different data
    - Especially good on GPUs(!)
- Task parallelism
    - Different cores perform different operations with (the same or)
      different data
- These can be combined

FIXME: add missing figure


# Parallel scaling

- Strong parallel scaling
   - Constant problem size
   - Execution time decreases in proportion to the increase in the number
     of cores
- Weak parallel scaling
   - Increasing problem size
   - Execution time remains constant when number of cores increases in
     proportion to the problem size

FIXME: add missing figure


# Amdahl's law

- Parallel programs often contain sequential parts
- *Amdahl's law* gives the maximum speed-up in the presence of
  non-parallelizable parts
- Main reason for limited strong scaling

FIXME: add missing figure


# Parallel computing concepts

- Load balance
    - Distribution of workload to different cores
- Parallel overhead
    - Additional operations which are not present in serial calculation
    - Synchronization, redundant computations, communications


# Summary

- HPC throughout the ages -- performance through parellelism
- Programming GPUs
    - CUDA, OpenCL
    - Directive based methods, OpenACC -- well supported based approach
- Parallel computing concepts
