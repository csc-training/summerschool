---
title:  Application design
event:  CSC Summer School in High-Performance Computing 2023
lang:   en
---

# Why develop software? 

- To do science
- To create a product

- Supercomputing platforms enable investigating bigger and more complex problems

<div class=column>
    
- **Do science**
    - Scientific articles
    - Method-oriented articles presenting code and methods
    - Speedup other standard scientific procedures, reducing also their costs
</div>

<div class=column>
    
- **Code as a product**
    - Prestige and fame
    - Gateway into projects, collaborations
    - Citations, co-authorships
    - Work on the bleeding edge

</div>


# Starting position

- New code or existing project / rewrite of old code?
    - How much effort do you have at your disposal?

- **Questions**: your software project?

# ICON

<div class=column>
- Icosahedral Nonhydrostatic Weather and Climate Model
- Closed source, developed my several meteorology institutes in Europe
    - DWD, MPI-M, MeteoSwiss, ...
    - ~200 developers
- 1 600 000 lines of modern Fortran
- Stencil type operations
    - Memory bound
- MPI + OpenMP
- GPUs with OpenACC (+CUDA/HIP)
  
</div>
<div class=column>
    
<!-- Image source https://code.mpimet.mpg.de/projects/iconpublic -->
![](images/r2b02_europe.png){.center width=100%}

</div>

# LiGen
<div class=column>
- High Throughput virtual screening Application
- C++ + GPU (CUDA/SYCL)
    - MPI + std::thread + offloading
    - Property of Dompe' Farmaceutici
    - Found a compound active against COVID (paucisymptomatic)
</div>
<div class=column>
    
![<span style=" font-size:0.5em;">image credits: https://exscalate.com/our-approach/exscalate-platform </span>](images/ligen.svg){.center width=50%}

</div>


# Starting position: Cases ICON & LiGen

<div class=column>
    
ICON

- Rewrite of earlier weather model in early 2000s
- Next generation modeling system capable of global simulations down to 1km resolutions
- Operational forecasts since 2015
- GPU porting started in ~2015

</div>

<div class=column>
    
LiGen

- Existing code with basic features mostly working
- Bad scaling paradigm
- Complete rewrite of the application to use advanced c++ features
    - Data flow design
    - GPU acceleration added
</div>


# Design model

- Development is not only about physics and numerics
    - Also about **how** you do it
- Instead of "Just code" it is advantageous to plan a little too!
    - Also think about future possible extensions!


# Parallelization strategies

- Planning includes thinking what is the target platform
- Target machines: laptops, small clusters, supercomputers
    - OpenMP, MPI, MPI+OpenMP, GPUs
- From shared memory to distributed memory machines
    - Keep in mind that most machines are distributed memory systems = MPI
- Moving from <1000 cores to >10k cores
    - Parallellization strategies need to be considered
    - Non-blocking, avoiding global calls,...
- Accelerators
    - GPUs have their own tricks and quirks

# Parallelization strategies 

- Going **BIG** -> GPUs are mandatory
- But not all HPC needs to be exascale
    - Size is not a goal in itself

# Case ICON: Parallellization

- Domain decomposition of the horizontal grid
- Halo exchange with `Isend` / `Irecv`
- Asynchronous I/O with special I/O processes
    - One sided MPI

# Case ICON: GPU porting

- Most of the computations on GPU
- Data copied to the GPU in the start of the simulation
- Loops parallelized with OpenACC directives
- GPU aware MPI communication

# Case LiGen: Parallellization

<div class=column>
    
- Embarassingly parallel (or data parallel)
- One process per node
- Internal pipeline
    - Thread balancing is a variable to explore!
        - Find the bottleneck, increase workers, repeat

- No need to communicate between nodes
    - MPI used only for IO operations

</div>

<div class=column>
<br />
<br />
<br />

![ <span style=" font-size:0.5em;">   image credits: https://arxiv.org/pdf/2110.11644.pdf </span> ](images/docker-ht-pipeline.jpg){.center width=100%}


</div>

# Case LiGen: GPU parallelization

<div class=column>
- Traditional approach: distribute computation
    - Latency oriented: process a single data as fast as possible

- New solution: batch of data
    - Throughput oriented: kernel is slower, but can handle much more data
    - If enough data are available, throughput is increased
    - Possible only if amount of memory per data required is small
</div>


<div class=column>

![](images/latency.jpg){.justify width=90%}

<br />

![ <span style=" font-size:0.5em;">images credits: https://arxiv.org/pdf/2209.05069.pdf </span> ](images/batch.jpg){.justify width=90%}

</div>

# Programming languages

- Selection of languages
    - Performance oriented languages (low level)
    - Programmability oriented languages (high level)
    - Mix
        - Best of both worlds
        - Low-level languages for costly functions
        - High-level languages for main functions

# Low level languages

- Direct control over memory
- Most common are C, C++, Fortran
- GPU has better support for C/C++, Fortran for kernels is not supported at all in HIP.

<div class=column>

- C++
    - std library for data structures
    - low level memory management (concept of data ownership, move semantics,...)
    - metaprogramming

</div>

<div class=column>

- Fortran
    - Good for number crunching
    - Good array syntax

</div>

# High level languages

- Python/Julia
    - Faster coding cycle and less error prone
    - Testing, debugging, and prototyping much easier
    - Built on top of high performance libraries (numpy, tensorflow,...)

- Combinations/suggestions
    - Python & C++ (PyBind11) for object-oriented programming
    - Julia & Fortran (native) for functional programming

# GPU programming approaches

- Directive based approaches: OpenACC and OpenMP
    - "standard" and "portable"
- Native low level languages: CUDA (NVIDIA) and HIP (AMD)
    - HIP supports in principle also NVIDIA devices
    - Fortran needs wrappers via C-bindings
- Performance portability frameworks: SYCL, Kokkos
    - Support only C++
- Standard language features: parallel C++, `do concurrent`
    - Rely on implicit data movements
    - Compiler support incomplete

# Modular code design: programming

- Good code is modular
    - Encapsulation 
    - Self-contained functions
    - No global variables, input what you need
- Modular code takes more time to design but is **a lot** easier to extend and understand

# Code design: tools

<div class=column>
    
- Avoid **not invented here** syndrome
- Leverage existing software and libraries
    - Libraries
        - Numerical (BLAS, solvers,...)
        - I/O
        - Parallelization
</div>

<div class=column>
    
- Caveats:
    - Is the lib still supported/updated?
    - Do you trust the source, is it widely used
    - Is there documentation
    - Does it support all the features
</div>

# Code design: development tools

- Software development is time consuming, many tools exist to help you in the process
- Build systems automate configuring and compiling
    - CMake
    - Make, Ninja
- The bigger your project is, the better is to rely on these automatic tools.
    - Setup can be painful
    
# Code design: development tools

- Debuggers
- Compilers
    - Compilers are not the same, compiler bugs are real!
    - Test your code with different compilers (gnu, clang, intel, cray,...)
- Linters (check coding style)

- **Questions**: Choices in your software and experiences about them?

# Case LiGen: Modular design

- Pipeline of stages with a common structure (input/output queues)
    - Easy to create new stages to support new functionalities
- Single interface for compute intensive backends
    - High level program structure separated by time consuming accelerated code
    - Different implementation for the accelerated code.
    - Backend characteristics (e.g data movement) are hidden from the rest of the application.


# Data design

- Data has to be "designed" too:
    - Use structures!
    - Think about the flow
    - How to distribute across the processes
    - GPU introduce more data related problems and opportunities:
        - Move between Host and Device
        - Preallocation
        - Overlapping computation with copy

# Case ICON: Data design

- Multiply nested data structures
- Need for deep copies for GPUs
- Fortran pointers used as "shortcuts"

# Case LiGen: Data design

- Relies a lot on c++ data ownership semantics (usage of move, refs, ...)
    - Avoid costly copies!

- GPU: 
    - Wrapper for memory to enable c++ RAII paradigm: we can forget about mallocs!
    - Preallocate for worst case: one malloc to process them all!
    - Double buffering

# IO Data formats

- Data formats
    - Not just plain text files/binary files
    - Platform-independent formats (HDF5, NetCDF, ...)
    - Metadata together with the data?
- Log files. Especially useful with HPC applications
- Standard formats
    - Your field might have some data standards 
- Remember also that large simulations produce lots of data
    - Storing "big data" is an issue

# Coding style

- Code readability comes first
- Consistency helps readability 
    - Indentation, how/when to have instructions longer than one line,...
    - Many editor have tools to help
    - There are exceptions!


# Summary 

- Software design is all about planning 
- Productivity
    - Modular design
    - Use existing libraries
    - Use & adopt design, community, and collaboration tools
    - Programming language and design selection


