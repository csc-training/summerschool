---
title:  IO libraries
event:  CSC Summer School in High-Performance Computing 2022
lang:   en
---


# I/O libraries

- How should HPC data be stored?
    - Large, complex, heterogeneous, esoteric, metadata ...
    - Parallel and random access
- Traditional relational databases poor fit
    - Cannot handle large objects
    - Many unnecessary features for HPC data
    - Poison for many parallel filesystems
- MPI-IO is efficient but relatively low level


# I/O libraries

- I/O libraries can produce files with standardized format
    - Portable files that can be manipulated with external software
    - Visualisation much easier for complicated cases
- Typically, I/O libraries support also metadata
    - Self-describing files, pays off in the long run
- Parallel I/O is typically build on top of MPI-IO
- Generic
    - **HDF5**, **NetCDF**, ADIOS, SIONlib
- Framework specific/builtin


# I/O libraries

- Consider the full data cycle
    - Raw IO performance is important but not the sole thing to focus on
- Standards and common formats in your field
    - Weather, PIC, AMR
- Lots of tuning still required, read the documentation!
    - Understanding the underlying system
- Future scalability
    - Dream big, but don't overengineer.
- File format and data format are different things.

# HDF5

- A data model, library, and file format for storing and managing
  multidimensional data
- Can store complex data objects and meta-data
- File format and files are *portable*
- Possibility for parallel I/O on top of MPI-IO
- Library provides Fortran and C/C++ API
    - Third party interfaces for Python, R, Java
    - Many tools can work with HDF5 files (Paraview, Matlab, ...)
- The HDF5 data model and library are complex

# HDF5 example dataset

![](img/hdf5.png)

# HDF5 hierarchy

![](img/hdf5_structure.png){.center width=70%}

# Exercise

_Discuss within the group_

- Are there any very common data formats in your field
- What does your process for the data look like?
    - Keeping track of what is what
    - Is the format hurting or helping your data-analysis
    - Will it be shared and with what kind of target audience?
- Estimate the size and bandwidth requirements of your input and output for current/future use cases
