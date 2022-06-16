---
title:  Introduction to GPUs in HPC
event:  CSC Summer School in High-Performance Computing 2022
lang:   en
---


# High-performance computing

<div class="column">

- High performance computing is fueled by ever increasing performance
- Increasing performance allows  breakthroughs in many major challenges that
  humankind faces today
- Not only hardware performance, algorithmic improvements have also added orders of magnitude of real performance

</div>

<div class="column">
![](img/top500-perf-dev.png)
</div>

# HPC through the ages

<div class="column" width=55%>
- Achieving performance has been based on various strategies throughout the years
    - Frequency, vectorization, multi-node, multi-core ...
    - Now performance is mostly limited by power consumption
- Accelerators provide compute resources based on a very high level of parallelism to reach
  high performance at low relative power consumption
</div>

<div class="column" width=43%>
![](img/microprocessor-trend-data.png)
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

# Why use them?
CPU vs Accelerator

![ <span style=" font-size:0.5em;">https://github.com/karlrupp/cpu-gpu-mic-comparison</span> ](img/comparison.png)


# Different design philosophies

<div class="column">

**CPU**
\
\

- General purpose
- Good for serial processing
- Great for task parallelism
- Low latency per thread
- Large area dedicated cache and control


</div>

<div class="column">

**GPU**
\
\

- Highly specialized for parallelism
- Good for parallel processing
- Great for data parallelism
- High-throughput
- Hundreds of floating-point execution units


</div>


# Lumi - Pre-exascale system in Finland

 ![](img/lumi.png){.center width=55%}


# Accelerator model today

<div class="column">
- Connected to CPUs via PCIe
- Local memory
    - Smaller than main memory (32 GB in Puhti)
    - Very high bandwidth (up to 900 GB/s)
    - Latency high compared to compute performance
- Data must be copied over the PCIe bus

</div>
<div class="column">
![](img/gpuConnect.png){}
![](img/gpu-bws.png){width=100%}
</div>

#  Heterogeneous Programming Model

- GPUs are co-processors to the CPU
- The CPU controls the work flow:
  - makes all the calls
  - allocates and deallocates the memory on GPUs
  - handles the data transfers between CPU and GPUs
  - delegates the highly-parallel work to the GPU
- The GPU code is executed by doing calls to functions (kernels) using thousands of threads running on thousands of cores
- The kernel calls are asynchronous

# Advance feature & Performance considerations

- Memory accesses:
   - data resides in the GPU memory; maximum performance is achieved when reading/writing is done in continuous manner
   - very fast on-chip memory can be used as a user programmable cache
- *Unified Virtual Addressing* provides unified view for all memory
- Asynchronous calls can be used to overlap transfers and computations.


# Challenges in using Accelerators

**Applicability**: Is your algorithm suitable for GPU?

**Programmability**: Is the programming effort acceptable?

**Portability**: Rapidly evolving ecosystem and incompatibilities between vendors.

**Availability**: Can you access a (large scale) system with GPUs?

**Scalability**: Can you scale the GPU software efficiently to several nodes?


# Using GPUs

<div class="column">
1. Use existing GPU applications
2. Use accelerated libraries
3. Directive based methods
    - **OpenMP**, OpenACC
4. Use native GPU language
    - CUDA, HIP, SYCL, Kokkos,...
</div>
<div class="column" width=40%>

Easier, but more limited

![](img/arrow.png){.center width=20% }

More difficult, but more opportunities

</div>




# Directive-based accelerator languages

- Annotating code to pinpoint accelerator-offloadable regions
- OpenACC
    - created in 2011, latest version is 3.1 (November 2020)
    - Mostly Nvidia
- OpenMP
    - Earlier only threading for CPUs
    - initial support for accelerators in 4.0 (2013), significant improvements & extensions in 4.5 (2015), 5.0 (2018), 5.1 (2020 and 5.2 (2021)

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


# GPUs @ CSC

- **Puhti-AI**: 80 nodes, total peak performance of 2.7 Petaflops
    - Four Nvidia V100 GPUs, two 20 cores Intel Xeon processors, 3.2 TB fast local storage, network connectivity of  200Gbps aggregate bandwidth  
- **Mahti-AI**: 24 nodes, total peak performance of 2. Petaflops
    - Four Nvidia A100 GPUs, two 64 cores AMD Epyc processors, 3.8 TB fast local storage,  network connectivity of  200Gbps aggregate bandwidth   
- **LUMI-G**: 2560 nodes, total peak performance of 500 Petaflops
    - Four AMD MI250X GPUs, one 64 cores AMD Epyc processors, 2x3 TB fast local storage, network connectivity of  800Gbps aggregate bandwidth

# Summary

- HPC throughout the ages -- performance through parallelism
- Programming GPUs
    - CUDA, HIP
    - Directive based methods
