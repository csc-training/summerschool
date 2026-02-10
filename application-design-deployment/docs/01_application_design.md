---
title:  Application design
event:  CSC Summer School in High-Performance Computing 2026
lang:   en
---

# Why develop software? 

<div class=column>
    
- **Do science**
    - Answer questions, solve problems
    - Scientific method:
        - Observations/measurements
        - Theoretical model
        - **Simulation**
    - Speedup scientific procedures and analysis, reducing also their costs
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
    - Number of developers may grow

- **Question**: What is your software project?


# Design model

- Development is not only about physics and numerics
    - Also about **how** you do it
- Instead of "Just code" it is advantageous to **plan** a little too!
    - Also think about future possible extensions!
- **Software engineering** has come up with lots of different development models
    - Waterfall, V-model, Agile models (Scrum etc.), ...
    - Also scientific software may benefit from formal development models


# Design considerations

- Parallelisation strategies
- Data design
- Programming languages
- Modularity
- I/O formats
- Documentation
- Testing


# Parallelisation strategies

- Planning includes thinking what is the target platform
- **Target machines**: laptops, small clusters, supercomputers
    - OpenMP, MPI, MPI+OpenMP, GPUs
- From shared memory to distributed memory machines
    - Keep in mind that most machines are distributed memory systems = MPI
- Moving from <1000 cores to >10k cores
    - Parallelisation strategies need to be considered
    - Non-blocking, avoiding global calls, ...
- **Accelerators**
    - GPUs have their own tricks and quirks

# Parallelisation strategies 

- Going **BIG** &rarr; GPUs are pretty mandatory these days
- But not all HPC needs to be exascale
    - Size is not a goal in itself


# Programming languages

- Selection of languages
    - **Performance** oriented languages (low level)
    - **Programmability** oriented languages (high level)
    - Mix
        - Best of both worlds
        - Low-level languages for costly functions
        - High-level languages for main functions


# Low level languages

- Direct control over memory
- Most common are **C, C++, Fortran**
    - Better support for GPU programming in C/C++, HIP does not support Fortran  kernels

<div class=column>

- C++
    - `std` library for data structures
    - low level memory management (concept of data ownership, move semantics, ...)
    - metaprogramming

</div>

<div class=column>

- Fortran
    - Good for number crunching
    - Good array syntax
    - Language semantics make optimisation easier for compilers

</div>


# High level languages

- Python/Julia
    - Faster coding cycle and less error prone
    - Testing, debugging, and prototyping much easier
    - Built on top of high performance libraries (numpy, tensorflow, ...)

- Combinations/suggestions
    - Python & C++ (PyBind11) for object-oriented programming
    - Julia & Fortran (native) for functional programming


# GPU programming approaches

- **Directive** based approaches: OpenACC and OpenMP
    - "standard" and "portable"
- **Native** low level languages: CUDA (NVIDIA) and HIP (AMD)
    - HIP supports in principle also NVIDIA devices
    - With HIP, Fortran needs wrappers via C-bindings
- **Performance portability** frameworks: SYCL, Kokkos
    - Support only C++
- **Standard language features**: parallel C++, `do concurrent`
    - Rely on implicit data movements
    - Compiler support incomplete


# Modular code design: programming

- Good code is **modular**
    - Encapsulation
    - Self-contained functions
    - No global variables, input what you need
- Modular code takes more time to design but is **a lot** easier to extend and understand


# Break for questions and breathing


# Version control

- Version control is the **single most important software development tool**
- Git is nowadays ubiquitous, Subversion (SVN) was more common but is less popular now
- Additional tools in web services (GitHub, GitLab, Bitbucket)
    - Forking
    - Issue tracking
    - Review of pull/merge requests
    - Wikis
    - Integrations
- Vlasiator: public [GitHub repository](https://github.com/fmihpc/vlasiator)


# Code design: tools

<div class=column>
    
- Avoid **not invented here** syndrome
- Leverage existing software and **libraries**
    - Numerical (BLAS, solvers, ...)
    - I/O
    - Parallelisation
    - Profiling/monitoring
</div>

<div class=column>
    
- Caveats:
    - Is the lib still supported/updated?
    - Do you trust the source, is it widely used?
    - Is there documentation?
    - Does it support all the features, can you extend it if needed?
</div>


# Code design: development tools

- Software development is time consuming, many **tools** exist to help you in the process
- Build systems automate **configuring and compiling**
    - CMake
    - GNU Autotools
    - Make, Ninja
- The bigger your project is, the better is to rely on these **automatic** tools
    - Setup can be painful

    
# Code design: development tools

- Debuggers
- Compilers
    - Compilers are not the same, compiler bugs are real!
    - Test your code with different compilers (gnu, clang, intel, cray, ...)
- Linters (check coding style)

- **Questions**: What development tools do you use? Do they make your work easier?


# Data design

- Data has to be "designed" too
- Use **structures**!
    - Note possible performance difference between structure of arrays vs.
      arrays of structures
- Think about the **flow**
- How to distribute the data 
- GPU introduce more data related problems and opportunities:
    - Memory copies between host and device
    - Preallocation, prefetching, overlapping computation with copy
    - GPU-aware MPI


# I/O Data formats

- **Data** formats
    - Not just plain text files/binary files
    - Platform-independent formats (HDF5, NetCDF, ...)
    - Metadata together with the data?
- **Log** files
- Standard formats
    - Your field might have some data standards 
- Remember also that large simulations produce lots of data
    - Storing "big data" is an issue
    - A global climate simulation can produce one PB in a day


# Coding style

- Code **readability** comes first
- **Consistency** helps readability 
    - Indentation, how/when to have instructions longer than one line, ...
    - Many editors have tools to help
    - There are exceptions!


# Documentation – TBD use Diátaxis framework
- In-code:
    - **Explanation** of what files, classes, functions are doing
    - Text and e.g. ascii-art explanations of complex parts
    - Can be formatted to build external documentation (e.g. `Doxygen`, `Sphinx`)
- Along with the code (wiki, manual)
    - **How to** contribute
    - How to install and use
    - How to analyse
    - How to cite


# Documentation
- **For whom** am I writing documentation? Think of:
    - You after vacation (did *I* write this?)
    - Who comes after your PhD (they *must* have had a good reason for writing it like this?!)
    - Future contributors (where to start? how do I contribute my optimised kernel to their repo?)
    - Future users (I could use this for my research, how does it work?)
- **What tools** do I use to support writing and deploying good documentation?


# Documentation – Diátaxis framework

![](images/diataxis_axes-of-needs.png){width=70%}

D. Procida, [Diátaxis documentation framework](https://diataxis.fr/) (CC-BY-SA 4.0)


# Documentation – Diátaxis framework

- What are the documentation users' needs?
- Practical knowledge vs. theoretical knowledge
- Acquiring knowledge vs. applying knowledge


# Testing
- **Unit** testing (does this function/solver/module work?)
- **Integration** testing (hopefully my new feature doesn't break everything?)
- **Verification** (does my code do what I designed it to do?)
- **Validation** (does my code do things as expected compared to theory/data?)

**Use automated tools to streamline as much testing as you can! Ensure your test coverage is adequate!**


# Conclusions 

- Software design is all about planning
- Productivity
    - Modular design
    - Use existing libraries
    - Use and integrate design, community, and collaboration tools
    - Programming language and design selection
- Re-/Usability
    - Not only for a single developer
    - Automation/standardisation where possible
    - Adopt practices and tools to ease the burden of a single person



