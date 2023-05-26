---
title:  Introduction to supercomputing
event:  CSC Summer School in High-Performance Computing 2023
lang:   en
---

# What is high-performance computing?

- Utilising computing power that is much larger than available in a typical desktop computer
- Performance of HPC systems (i.e. supercomputers) is often measured in floating point operations per second (flop/s)
    - For software, other measures can be more meaningful
- Currently, the most powerful system reaches > $10^{18}$ flop/s (1 Eflop / s)

# What is high-performance computing?

<!-- Copyright CSC -->
 ![](img/cray.png){.center width=30%}

# Top 500 list

<!-- Source: top500.org, Copyright 1993-2023 TOP500.org (c) -->
 ![](img/top_500.png){.center width=50%}

# What are supercomputers used for? {.section}

# General use cases

- Simulations of very different scales
    - From subatomic particles to cosmic scales
- Problems with very large datasets
- Complex computational problems
- Problems that are hard to experiment on
    - Simulations with decade-long timescales
- Very time consuming or even impossible to <br> solve on a standard computer

# Application areas

- Fundamental sciences such as particle physics and cosmology
- Climate, weather and earth sciences
- Life sciences and medicine
- Chemistry and material science
- Energy, e.g renewable sources and fusion research
- Engineering, e.g. manufacturing and infrastructure
- etc.

# Climate change

<div class=column>
- Simulating ice sheets, air pollutants, sea-level rise etc.
- Building short and long-term simulations
- Analyzing with different parameters for future predictions and possible solutions
- Modeling space weather
</div>
<div class=column>
![](img/climate_greenland.png){.center width=90%}
</div>

# Covid-19 fast track with Puhti

<div class=column>
- Modeling particles in airflows
- A large part of the calculations used for solving turbulent flow
- A third of Puhti was reserved for running the simulations
- The results have had an impact on e.g. ventilation instructions and the use of masks
</div>
<div class=column>
![](img/covid.png){.center width=100%}
</div>

# Gravitational waves

<div class=column>
- Computational modeling of sources of gravitational waves
- Identifying a phase transition of the Higgs boson “turning on” (10 picoseconds after Big Bang)
- Large simulations with over ten thousand CPU cores
- Experimental data from ESA's LISA satellite (Launch date 2037)
</div>
<div class=column>
![](img/grav.png){.center width=90%}
</div>

#  Topological superconductors

<div class=column>
- Topological superconductors are possible building blocks for qubits
- Based on an elusive quantum state of electrons in thin layers
- Electronic properties simulated with the density-functional theory
    - These confirm that experimentally measured signals are due to this special quantum state
</div>
<div class=column>
![](img/majorana.png){.center width=100%}
</div>

# Deep language model of Finnish

<div class=column>
- Web-scale Finnish language data together with very deep neural networks utilizing GPUs
- Open GPT-3 model ("Finnish ChatGPT")
</div>
<div class=column>
<!-- Source: Adobe Stock, CC BY-SA 3.0 -->
 ![](img/DeepFin.jpg){.center width=100%}
</div>

# Utilizing HPC in scientific research

<!-- Copyright CSC -->
 ![](img/sci.png){.center width=40%}

- **Goal for this school: everyone is able to write and modify HPC applications!**

# What are supercomputers made of? {.section}

# CPU frequency development

- Power consumption of CPU: $~f^3$

 ![](img/moore.png){.center width=45%}

# Parallel processing

- Modern (super)computers rely on parallel processing
- **Multiple** CPU cores & accelerators (GPUs)
    - `#`1 system has `~`600 000 cores and `~`40 000 GPUs
    - `#`2 system (CPU-only) has `~`8 000 000 cores
- Vectorization
    - A single instruction can process multiple data (SIMD)
- Pipelining
    - Core executes different parts of instructions in parallel


# Anatomy of a supercomputer

<!-- Copyright CSC -->
 ![](img/anatomy.svg){.center width=55%}

- Supercomputers consist of nodes connected by a high-speed network
    - Latency `~`1 µs, bandwidth `~`100 GB / s
- A node can contain several multicore CPUs and several GPUs
- Memory within the node is directly usable by all CPU cores
- GPUs have their own local memory

# Supercomputer autopsy – Lumi

 ![](img/lumi.png){.center width=50%}

# From laptop to Tier-0

<div class=column>

<!-- Copyright CSC -->
 ![](img/tier.png){.center width=80%}

</div>
<div class=column>
- The most fundamental difference between a small university cluster and Tier-0 supercomputer is the number of nodes
    - The interconnect in high end systems is often also more capable
</div>

# Cloud computing

- Cloud infrastructure is run on top of normal HPC system:
    - Shared memory nodes connected by network
- User obtains **virtual** machines
- Several providers offer also bare metal instances
- Infrastructure as a service (IaaS)
    - User has full freedom (and responsibility) of operating system and the whole software environment
- Platform as a service (PaaS)
    - User develops and runs software within the provided environment

# Cloud computing and HPC

- Suitability of cloud computing for HPC depends heavily on application
- Virtualization adds overhead especially for the networking
- Bare metal cloud with high-speed interconnects can provide similar performance as traditional cluster
- Moving data out from the cloud can be time-consuming (and have a monetary cost)
- Cost-effectiveness of cloud depends heavily on the use case

# Containers

TO BE ADDED

# Parallel computing concepts {.section}

# Computing in parallel

- Parallel computing
    - A problem is split into smaller subtasks
    - Multiple subtasks are processed simultaneously using multiple cores

<!-- Copyright CSC -->
 ![](img/compp.svg){.center width=40%}

# Types of parallel problems

- Tightly coupled
    - Lots of interaction between subtasks
    - Weather simulation
    - Low latency, high speed interconnect is essential
- Embarrassingly parallel
    - Very little (or no) interaction between subtasks
    - Sequence alignment queries for multiple independent sequences in bioinformatics


# Exposing parallelism

<div class=column>
- Data parallelism
    - Data is distributed across cores
    - Each core performs simultaneously (nearly) identical operations with different data
    - Cores may need to interact with each other, e.g. exchange information about data on domain boundaries
</div>
<div class=column>

<!-- Copyright CSC -->
 ![](img/eparallel.svg){.center width=80%}

</div>

# Exposing parallelism

- Task farm (master / worker)

<!-- Copyright CSC -->
 ![](img/farm.svg){.center width=60%}

<br>

- Master sends tasks to workers and receives results
- There are normally more tasks than workers, and tasks are assigned dynamically

# Parallel scaling

<div class=column>
- Strong parallel scaling
    - Constant problem size
    - Execution time decreases in proportion to the increase in the number of cores
- Weak parallel scaling
    - Increasing problem size
    - Execution time remains constant when number of cores increases in proportion to the problem size
</div>
<div class=column>

<!-- Copyright CSC -->
 ![](img/scaling.png){.center width=80%}

</div>

# What limits parallel scaling

<div width=55% class=column>
- Load imbalance
    - Variation in workload over different cores
- Parallel overheads
    - Additional operations which are not present in serial calculation
    - Synchronization, redundant computations, communications
- Amdahl’s law: the fraction of non-parallelizable parts limits maximum speedup
</div>
<div width=40% class=column>
  ![](img/AmdahlsLaw.svg){.right width=100%}
</div>


# Parallel programming {.section}

# Programming languages

- The de-facto standard programming languages in HPC are (still!)
  C/C++ and Fortran
- Higher level languages like Python and Julia are gaining popularity
    - Often computationally intensive parts are still written in C/C++
      or Fortran
- Low level GPU programming with CUDA or HIP
- For some applications there are high-level frameworks with
  interfaces to multiple languages
    - SYCL, Kokkos, PETSc, Trilinos
    - TensorFlow, PyTorch for deep learning

# Parallel programming models

- Parallel execution is based on threads or processes (or both) which run at the same time on different CPU cores
- Processes
    - Interaction is based on exchanging messages between processes
    - MPI (Message passing interface)
- Threads
    - Interaction is based on shared memory, i.e. each thread can access directly other threads data
    - OpenMP, pthreads

# Parallel programming models

<!-- Copyright CSC -->
 ![](img/processes-threads.svg){.center width=80%}
<div class=column>
**MPI: Processes**

- Independent execution units
- MPI launches N processes at application startup
- Works over multiple nodes
</div>
<div class=column>

**OpenMP: Threads**

- Threads share memory space
- Threads are created and destroyed  (parallel regions)
- Limited to a single node

</div>

# GPU programming models

- GPUs are co-processors to the CPU
- CPU controls the work flow:
  - *offloads* computations to GPU by launching *kernels*
  - allocates and deallocates the memory on GPUs
  - handles the data transfers between CPU and GPUs
- GPU kernels run multiple threads
    - Typically much more threads than "GPU cores"
- When using multiple GPUs, CPU runs multiple processes (MPI) or multiple threads (OpenMP)

# GPU programming models

![](img/gpu-offload.svg){.center width=60%}

# Parallel programming models

![](img/anatomy.svg){.center width=100%}

# Future of High-performance computing {.section}

# Quantum computing

<div class=column>
- Quantum computers can solve certain types of problems exponentially faster than classical computers
- General purpose quantum computer is still far away
- Use cases still largely experimental and hypothetical
- Hybrid approaches
</div>
<div class=column>
![](img/quantum.png){.center width=50%}
</div>


# Post-Exascale challenges

- Performance of supercomputers has increased exponentially for a long time
- However, there are still challenges in continuing onwards from exascale supercomputers ($> 1 \times 10^{18}$ flop/s)
    - Power consumption: current `#`1 energy efficient system requires `~`20 MW for exascale performances
    - Cost & Maintaining: Global chip shortage
    - Application scalability: how to program for 100 000 GPUs / 100 000 000 cores?

