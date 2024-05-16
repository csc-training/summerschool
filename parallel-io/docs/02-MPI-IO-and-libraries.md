---
title:  MPI-IO and I/O libraries
event:  CSC Summer School in High-Performance Computing 2024
lang:   en
---

# MPI-IO {.section}

# MPI-IO

- MPI-IO stands for Message Passing Interface I/O.
- It is a standard low-level interface for performing I/O operations in parallel computing.
- MPI-IO allows processes in a parallel application to read from and write to files simultaneously.
- It provides collective and non-collective I/O operations for efficient data access and manipulation.

# Key Features of MPI-IO

- Collective I/O: Enables a group of processes to perform I/O operations as a single unit, improving performance by reducing communication overhead.
- Independent I/O: Allows each process to perform I/O operations independently, suitable for irregular or non-contiguous data access patterns.
- File Views: Supports defining non-contiguous regions of data in a file and performing I/O operations on those regions.
- Data Types: Supports complex data types, allowing efficient I/O operations on structured data.

# Key Features of MPI-IO (cont.)

- Shared File Pointers: Enables multiple processes to share a common file pointer, ensuring coordinated access to the file.
- Error Handling: Provides error reporting and handling mechanisms to facilitate robust I/O operations.

# Benefits of MPI-IO

- Parallelism: MPI-IO enables multiple processes to access files concurrently, improving performance in parallel applications.
- Flexibility: It supports various I/O patterns, such as independent I/O, collective I/O, and one-sided I/O.
- Scalability: MPI-IO can efficiently scale to large numbers of processes and handle large datasets.
- Portability: It is a portable standard supported by different MPI implementations.

# I/O libraries

- How should HPC data be stored?
    - Large, complex, heterogeneous, esoteric, metadata ...
    - Parallel and random access
- Traditional relational databases poor fit
    - Cannot handle large objects
    - Many unnecessary features for HPC data
    - Poison for many parallel filesystems
- MPI-IO is efficient but relatively low level

# Why use I/O libraries?

- I/O libraries can produce files with standardized format
    - Portable files that can be manipulated with external software
    - Visualisation much easier for complicated cases
- Typically, I/O libraries support also metadata
    - Self-describing files, pays off in the long run
- Parallel I/O is typically build on top of MPI-IO
- Generic
    - **HDF5**, **NetCDF**, ADIOS, SIONlib
- Framework/application domain specific
    - Weather, PIC, astronomy...

# Using I/O libraries

- Consider the full data cycle
    - Raw IO performance is important but not the sole thing to focus on
- Lots of tuning still required, read the documentation!
    - Understanding the underlying system
- Future scalability
    - Dream big, but don't overengineer.
