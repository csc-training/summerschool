---
title:  Introduction to <br>high-performance computing
event:  CSC Summer School in High-Performance Computing 2025
lang:   en
---

# Outline

*High-performance computing (HPC) is the use of supercomputers and<br>computer clusters to solve advanced computation problems.*<br>
− [Wikipedia](https://en.wikipedia.org/wiki/High-performance_computing)
<br>
<br>

- Why to use supercomputers?
- What are supercomputers?
- Current trends in high-performance computing


# Why to use supercomputers? {.section}

# Supercomputer application areas

- Supercomputers enable solving computation problems that are impossible or too slow to solve on a standard computer
- Application areas are everywhere
  - Climate, weather, and earth sciences
  - Chemistry and material science
  - Particle physics and cosmology
  - Life sciences and medicine
  - Renewable energy and fusion research
  - Engineering
  - Large-scale AI
  - ...

# Climate and weather research

<div class=column>
- Simulating ice sheets, air pollutants, sea-level rise etc.
- Building short and long-term simulations
- Simulated areas can span over countries or continents
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

# Large-scale AI

<div class=column>
- [Training LLMs](https://lumi-supercomputer.eu/scaling-the-pre-training-of-large-language-models-of-100b-parameters-to-thousands-of-amd-mi250x-gpus-on-lumi/)
  - [OpenEuroLLM](https://lumi-supercomputer.eu/open-euro-llm/)
</div>
<div class=column>
<!-- Source: Adobe Stock, CC BY-SA 3.0 -->
 ![](img/DeepFin.jpg){.center width=100%}
</div>


# What are supercomputers? {.section}

# Early supercomputers

<div class=column>
- Specialized hardware, substantially faster than other computers at the time
- Examples
  - CDC 6600 (1964):<br>
    CPU 10 MHz, 982 kB RAM
  - Cray-1 (1975):<br>
    CPU 80 MHz, 8.4 MB RAM
  - Cray X-MP (1982):<br>
    CPU 4x ~110 MHz, 128 MB RAM
</div>
<div class=column>
<center>
![](img/cray-x-mp.jpg){.center width=100%}
Cray X-MP at CSC (1989)<br>
Image: CSC Archives
</center>
</div>

# Current supercomputers

<div class=column>
- Large computer clusters
- A lot of standard, high-end server hardware connected to each other
- High-speed interconnect between compute nodes
- Example: LUMI
  - 2064 CPU nodes<br>(~260,000 CPU cores in total)
  - 2978 GPU nodes<br>(~12,000 MI250X GPUs in total)
</div>
<div class=column>
<center>
![](img/lumi.jpg){.center width=100%}
EuroHPC JU LUMI (HPE Cray EX) <br>at CSC (2021–)
</center>
</div>

# Anatomy of a supercomputer

<!-- Copyright CSC -->
![](img/cluster_diagram.svg){.center width=100%}

# Closer look on supercomputer nodes

<!-- Copyright CSC -->
 ![](img/anatomy.svg){.center width=55%}

- Supercomputers consist of nodes connected by a high-speed network
  - Latency `~`1 µs, bandwidth `~`100 GB / s
- A node can contain several multicore CPUs and several GPUs
- Permanent storage is accessed via network (shared resource among *all* users)

# Computer clusters and supercomputers

- A computer cluster is a set of computers connected to each other
- The cluster is visible for the user as a single system
- Most current supercomputers are very large computer clusters

<br>
<br>
<center>
**A supercomputer is not a single "super fast" processor,<br>but a set of hundreds of thousands (or more) processing units<br>that *can be programmed* to work together**<br><br>
&rarr; *Parallel algorithms* are needed to utilize supercomputer
</center>

# Parallel processing

- Modern supercomputers (and regular computers) rely on parallel processing
- **Multiple** CPU cores & accelerators (GPUs)
  - LUMI-G has ~12,000 GPUs together with ~190,000 CPU cores
  - LUMI-C has ~260,000 CPU cores
- Vectorization
  - A single instruction can process multiple data (SIMD)
- Instruction level parallelism and pipelining
  - Core executes multiple instructions in parallel (superscalar execution)
  - Core executes different parts of instructions in parallel

# Why parallel processing?

- Power consumption of CPU: $~f^3$

![](img/moore.png){.center width=45%}


# Performance of supercomputers

<div class=column style=width:59%>
- Performance is often measured in floating point operations per second (flop/s)
- [LUMI (GPU partition)](https://top500.org/system/180048/)
  - Linpack: 380 PFlop/s (FP64)
  - (Theoretical: 530 PFlop/s (FP64))
  - Power: 7.1 MW &rarr; 54 GFlop/s/W (FP64)
- [LUMI (CPU partition)](https://top500.org/system/180045/)
  - Linpack: 6.3 PFlop/s (FP64)
  - (Theoretical: 7.6 PFlop/s (FP64))
  - Power: 1.2 MW &rarr; 5.3 GFlop/s/W (FP64)
</div>
<div class=column style=width:39%>
<center>
![](img/top_500.png){.center width=100%}
Image: [top500.org](https://top500.org/)
</center>
</div>


# From laptop to Tier-0

<div class=column>

<!-- Copyright CSC -->
 ![](img/tier.svg){.center width=80%}

</div>
<div class=column>
- The most fundamental difference between a small university cluster and Tier-0 supercomputer is the number of nodes
  - The interconnect in supercomputers is often also more capable
</div>


# Future of high-performance computing {.section}

# GPUs are becoming the norm

<center>
![](img/top500_nov21.svg){.center width=65%}

- Top500 supercomputers grouped by the accelator type (Nov 2021 list)
</center>

# GPUs are becoming the norm

<center>
![](img/top500_nov22.svg){.center width=65%}

- Top500 supercomputers grouped by the accelator type (Nov 2022 list)
</center>

# GPUs are becoming the norm

<center>
![](img/top500_nov24.svg){.center width=65%}

- Top500 supercomputers grouped by the accelator type (Nov 2024 list)
</center>


# Post-exascale challenges

- Performance of supercomputers has increased exponentially for a long time
- However, there are still challenges in continuing onwards from exascale supercomputers ($> 1 \times 10^{18}$ Flop/s)
  - Power consumption: the exascale supercomputers consume >20 MW
  - Manufacturing: Transistor sizes being extremely small
  - Application scalability: how to program for 100,000 GPUs / 100,000,000 CPU cores?


# Cloud computing vs high-performance computing

- Many similarities
  - Both enable remote execution of computations on a computer system with more computing power
- Many differences
  - Different software and computing environment
  - HPC: maximal performance (incl. high-end, tuned hardware and software) &rarr; for tightly coupled workloads
  - Cloud: flexibility, fault-tolerance &rarr; for loosely coupled workloads
- HPC-as-a-Service changing the "traditional" way to use HPC
  - [HEAppE](https://heappe.eu), [LEXIS](https://lexis-project.eu), ...


# Quantum computing

<div class=column>
- Solving certain types of problems exponentially faster than classical computers
- General-purpose quantum computer is still far away
- Use cases still largely experimental
- Hybrid approaches combining classical and quantum computing
  - Both are needed
</div>
<div class=column>
![](img/quantum.png){.center width=50%}
</div>


# Summary {.section}

# Utilizing HPC in scientific research

<!-- Copyright CSC -->
![](img/sci.png){.center width=40%}

- **Goal for this school: everyone is able to write and modify HPC applications!**

# Summary

- HPC is essential in many fields of research
- Supercomputers base their performance on extreme parallelism
  - Including GPUs nowadays
- Cloud computing as an option for accessing HPC resources
