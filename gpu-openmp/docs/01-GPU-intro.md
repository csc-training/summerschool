---
title:  Introduction to GPUs in HPC
event:  CSC Summer School in High-Performance Computing 2023
lang:   en
---


# High-performance computing

<div class="column" width= 55%>

- High performance computing is fueled by ever increasing performance
- Increasing performance allows  breakthroughs in many major challenges that humankind faces today
- Not only hardware performance, algorithmic improvements have also helped a lot

</div>

<div class="column" width=43%>
![](img/top500-perf-dev.png)
</div>

# HPC through the ages

<div class="column" width=55%>
- Various strategies to achieve performances through the years:
    - Frequency, vectorization, multi-node, multi-core ...
    - Now performance is mostly limited by power consumption
- Accelerators provide compute resources with high parallelism to reach high performance at low relative power consumption
</div>

<div class="column" width=43%>
![](img/microprocessor-trend-data_nocapt.png)
![](img/caption.png)
</div>


# Accelerators

- Specialized parallel hardware for floating point operations
    - Co-processors for traditional CPUs
    - Based on highly parallel architectures
    - Graphics processing units (GPU) have been the most common
      accelerators during the last few years
- Promises
    - Very high performance per node
- Usually major rewrites of programs required


# Flynn's taxonomy

- Proposed in 1966 to classify compute units

<div class="column" style="width:44%; padding-left:2em">
<p>
![](img/sisd.png){height=200px}<br>
Single Instruction Single Data<br>
<br>
![](img/simd.png){height=200px}<br>
Single Instruction Multiple Data
</p>
</div>

<div class="column" style="width:48%">
<p>
![](img/misd.png){height=200px}<br>
Multiple Instructions Single Data<br>
<br>
![](img/mimd.png){height=200px}<br>
Multiple Instructions Multiple Data
</p>
</div>


# Flynn's taxonomy

- Proposed in 1966 to classify compute units

<div class="column" style="width:44%; padding-left:2em">
<p>
![](img/sisd.png){height=200px}<br>
CPU - 1 core<br>
<br>
![](img/simd.png){height=200px}<br>
GPU
</p>
</div>

<div class="column" style="width:48%">
<p>
![](img/misd.png){height=200px}<br>
Not really used<br>
<br>
![](img/mimd.png){height=200px}<br>
CPU - manycores
</p>


# Different design philosophies: CPU

<div class="column">

**CPU**

- General purpose
- Low latency per thread
- Large area dedicated to caches and control
    - Good for control-flow
    - Great for task parallelism (MIMD)
- Less silicon dedicated at Arithmetic-Logic Units (ALU)
    - Bad with parallel execution


</div>

<div class="column">
![ <span style=" font-size:0.5em;">image credits: Nemez @GPUsAreMagic https://nemez.net/die/RocketLake</span> ](img/core_die_shot.jpg){width=57%}

</div>


# Different design philosophies: GPU
<div class="column">

**GPU**

- Most of the silicon dedicated to ALUs
    - Hundreds of floating-point execution units
    - Highly specialized for parallelism
- Great for data parallelism
- High-throughput
- Bad at control-flow processing


</div>


<div class="column">
![ <span style=" font-size:0.5em;">image credits: Nemez @GPUsAreMagic https://nemez.net/die/Ampere</span> ](img/gpu_die_shot.jpg){width=70%}

</div>


# Lumi - Pre-exascale system in Finland

 ![](img/lumi.png){.center width=50%}


# Accelerator model in LUMI 

<div class="column">
- GPU is connected to CPUs via Infinity Fabric
- Local memory in GPU
    - Smaller than main memory (128 GB)
    - Very high bandwidth (up to 3200 GB/s)
    - Latency high compared to compute performance
- Data must be copied from CPU to GPU over the Infinity Fabric

</div>
<div class="column" >
![](img/lumi-interconnectimgonly.png){width=70%}
![](img/lumi-interconnectlegend.png){width=70%}
</div>

#  Heterogeneous Programming Model

- GPUs are co-processors to the CPU
- CPU controls the work flow:
  - *offloads* computations to GPU by launching *kernels*
  - allocates and deallocates the memory on GPUs
  - handles the data transfers between CPU and GPUs
- CPU and GPU can work concurrently
   - kernel launches are normally asynchronous

# GPU architecture

<div class="column">
- Designed for running tens of thousands of threads simultaneously on
  thousands of cores
- Very small penalty for switching threads
- Running large amounts of threads hides memory access penalties
- Very expensive to synchronize all threads
</div>

<div class="column">
![](img/mi100-architecture.png)
<small>AMD Instinct MI100 architecture (source: AMD)</small>
</div>


# Advance features & Performance considerations

- Memory accesses:
   - data resides in the GPU memory; maximum performance is achieved when reading/writing is done in continuous blocks
   - very fast on-chip memory can be used as a user programmable cache
- *Unified Virtual Addressing* provides unified view for all memory
- Asynchronous calls can be used to overlap transfers and computations


# Challenges in using Accelerators

**Applicability**: Is your algorithm suitable for GPU?

**Programmability**: Is the programming effort acceptable?

**Portability**: Rapidly evolving ecosystem and incompatibilities between vendors.

**Availability**: Can you access a (large scale) system with GPUs?

**Scalability**: Can you scale the GPU software efficiently to several nodes?


# Using GPUs

1. Use existing GPU applications
2. Use accelerated libraries
3. Directive based methods
    - **OpenMP**, OpenACC
4. Use native GPU language
    - CUDA, HIP

# Directive-based accelerator languages

- Annotating code to pinpoint accelerator-offloadable regions
- OpenACC
    - created in 2011, latest version is 3.1 (November 2020)
    - mostly Nvidia
- OpenMP
    - earlier only threading for CPUs
    - initial support for accelerators in 4.0 (2013), significant improvements & extensions in 4.5 (2015), 5.0 (2018), 5.1 (2020) and 5.2 (2021)

- Focus on optimizing productivity
- Reasonable performance with quite limited effort, but not guaranteed



# Native GPU code: HIP / CUDA

- CUDA
    - has been the *de facto* standard for native GPU code for years
    - extensive set of optimised libraries available
    - custom syntax (extension of C++) supported only by CUDA compilers
    - support only for NVIDIA devices
- HIP
    - AMD effort to offer a common programming interface that works on
      both CUDA and ROCm devices
    - standard C++ syntax, uses nvcc/hcc compiler in the background
    - almost a one-on-one clone of CUDA from the user perspective
    - ecosystem is new and developing fast

# Using GPUs

![](img/gpu_approaches.png){.center width=70%}

# Directive languages and performances

- "Write once, run everywhere"
    - It is true that you get portability
    - It is *not* true that you get *performance* portability

- It is possible to optimize code for performance on GPU!
    - It will however be probably slower on the CPU




# GPUs @ CSC

- **Puhti-AI**: 80 nodes, total peak performance of 2.7 Petaflops
    - Four Nvidia V100 GPUs, two 20 cores Intel Xeon processors, 3.2 TB fast local storage, network connectivity of  200Gbps aggregate bandwidth  
- **Mahti-AI**: 24 nodes, total peak performance of 2. Petaflops
    - Four Nvidia A100 GPUs, two 64 cores AMD Epyc processors, 3.8 TB fast local storage,  network connectivity of  200Gbps aggregate bandwidth   
- **LUMI-G**: 2560 nodes, total peak performance of 500 Petaflops
    - Four AMD MI250X GPUs, one 64 cores AMD Epyc processors, 2x3 TB fast local storage, network connectivity of  800Gbps aggregate bandwidth

# Summary

- GPUs provide significant speed ups for certain applications
- GPUs are co-processors to CPUs
   - CPU offloads computations and manages memory
- High amount of parallelism required for efficient utilization of GPUs
- Programming GPUs
    - Directive based methods
    - CUDA, HIP
