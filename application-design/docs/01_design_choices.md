---
title:  Application design
author: CSC Summerschool
date:   2019-06
lang:   en
---

# Design choices {.section}

# Introducing case examples
- PlasmaBox
- GPAW
- Science with codes

# Why develop software? To do science, create a product, ...
- New code or existing project / rewrite of old code
- **Questions**: your software project
- Cases: Plasmabox (new code), GPAW (existing code)

- do science
    - scientific articles
    - method-oriented articles presenting code and methods
- code as a product
    - prestige and fame
    - gateway into projects, collaborations
    - citations, co-authorships
    - work on the bleeding edge

# Design model
- Science is (or should be) unpredictable? 
- let's learn from CS ppl
- agile development
- agile
    - umbrella term for different SW methods
    - divide work into small tasks, define short working period, review, repeat
    - SW evolves through collaboration 
    - focused on iterative and incremental dev'ing
        - quick prototyping
        - supports continuous publication
        - analysis, coding, testing, etc. never end

# Parallelization strategies
- Target machines, 
    - MPI, MPI+OpenMP, GPUs
- from shared memory to distributed memory machines
    - mpi
- from 1000 cores to >10k cores
    - non-blocking; avoiding global calls,...
- accelerators
    - gpus, what machine?

# Programming languages
- Coding style and modularity
- Tools: compilers, build systems, external libraries, debuggers, ....
- **Questions**: choices in your software and experiences about them
- Cases: Plasmabox, GPAW

- leverage existing software and libraries
    - libraries
        - numerical (blas, solvers)
        - I/O
        - parallelization
    - models
        - physical models
    - frameworks
        - plug your model
        - couplers that can be used to bind different models
        - Petsc, Trilinos, BoxLib++, Charm++, ARMReX,...
    - avoid not invented here syndrome
    - caveats:
        - is the lib still supported/updated?
        - do you trust the source, is it widely used
        - is there documentation
        - does it support all the features
- languages
    - most common are C, C++, Fortran
        - mostly matter of taste
    - C++ more full features object-oriented features and many more data structures (maps, lists, etc.)
    - Fortran is really for number crunching, good array syntax
    - Python: 
        - faster coding cycle and less error prone
        - testing, debugging, and prototyping much easier
- Different languages can be interfaced together
    - best of both worlds
    - low level languages (C, C++, Fortran) for costly functions
    - high-level languages (Python, Julia, R) for main functions
    - combinations
        - Python & C++ (PyBind11) for object-oriented programming
        - Julia & Fortran (native) for functional programming
- modernization
    - replicate main loop using high-level lang
    - interface existing functions into it
    - start rewriting & redesigning old functions that you really need
    - always a working version of the code

# Data formats
- log files, "big" data
- standard formats
    - e.g., PML



# Summary 
- learn from computer scientist
- productivity
    - use existing libraries
    - use & adopt community and collaboration tools
    - programming language and design selection
- open vs closed source


