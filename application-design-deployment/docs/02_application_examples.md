---
title:  Application examples 
event:  CSC Summer School in High-Performance Computing 2026
lang:   en
---


# Today's real-world case: Vlasiator

<div class=column style="width:65%">
- Hybrid-Vlasov model of magnetised space plasma physics
- [Open source](https://github.com/fmihpc/vlasiator), led by University of Helsinki
- \>60 000 lines of modern C++
- Semi-Lagrangian solver on Cartesian grids
- 6D: 3D space, 3V velocity
    - ghost updates in every direction, every step
    - **Lots** of memory
- MPI + OpenMP + SIMD, Cuda/HIP for GPUs
</div>

<div class=column style="width:33%">    
![ M. Alho / Vlasiator team](images/vlasiator_Alho_illustration.png){width=100%}
</div>


# Starting position: Case Vlasiator (ca. 2008)

- Science case: apply algorithm with "**more physics**" (kinetic phenomena)
    - Previous state of the art: fluid-based approaches
- Problem: "**too heavy**" for current HPC systems
- Strategy: start developing now what runs on "next year's" **cutting-edge HPC** systems
- Global (2D) simulations of the Earth's magnetosphere since 2013, 3D since 2020
- **GPU** porting started earnestly around 2021
    - Before that, insufficient memory and too low host &harr; device bandwidth


# Case Vlasiator: Parallelisation

<div class=column style="width:57%">
- Domain decomposition (**MPI**) of position-space grid
- OpenMP **threading** for position and velocity spaces
- **Vectorization** of core solvers
- Parallel MPI-IO with own library
    - Adapted for 6D data
- Good parallel scalability
    - Current record: 1452 nodes / 185 856 cores on LUMI-C
</div>

<div class=column style="width:40%">
![<span style=" font-size:0.5em;">   [\[Kotipalo et al., 2025\]](https://doi.org/10.48550/arXiv.2505.20908)](images/vlasiator_Kotipalo_load_balance.png){.center width=100%}
</div>


# Case Vlasiator: GPU porting strategy

- Initial exploration/hackathons e.g. using OpenACC
- Decision to use **unified memory**
- Piece-by-piece porting to GPU
- Performance analysis, to guide **re-design** effort
- **Re-writing** to consolidate GPU version
- Performance on-par with CPU
- Effort: > 4 person-years


# Case Vlasiator: GPU porting

- Most of the computations on GPU
- Data resides in **unified memory**, page faults successfully minimised
- CUDA/HIP kernels for portability
    - **Algorithms redesigned** to operate on at least 64 elements at a time
    - Extensive **kernel merging**
- **GPU-aware MPI communication**
- Custom hybrid hashmap and vector libraries [\[Papadakis et al., Front. Comput. Sci., 2024\]](https://doi.org/10.3389/fcomp.2024.1407365)


# Case Vlasiator: Modular design

- **Object-oriented programming** to manage e.g. boundary conditions
    - Easy to create new boundaries or initial conditions
- Separate **what** to do (e.g. physics) from **how** to do (e.g. optimal implementation of loops)
    - Interfaces to different backends (CUDA kernel, OpenMP threading, ...)
    - Use of class inheritance/templates/lambdas to reduce code duplication


# Case Vlasiator: Data design

- Adaptively refined **3D spatial mesh**
    - Array of spatial variables
    - (Hybrid) hashmap of (sparse, dynamic) **3V velocity mesh**
    - 3D-3V used for (computationally heavy) propagation of plasma
- Uniform, highest-resolution 3D spatial mesh
    - Arrays of electromagnetic field variables
    - Used for (computationally lighter) EM field propagation


# Case Vlasiator: I/O data formats

- Input data (run parameters): command line arguments or **ascii** configuration file
- Logfile with simulation progress, memory monitoring, etc. in ascii
- Output data in custom format with library using **fast, parallel MPI-IO** (10s–100s GiB/s; reduces redundancy of stored metadata for 3D-3V data compared to industry-standard formats)
- Checkpoint/restart files in same custom format
- Can be 10 TB per file
    - Need to consider post-processing and long-term storage, collaboration



