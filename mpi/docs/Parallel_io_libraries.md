---
title: Parallel I/O libraries 
author: CSC Summerschool
date:   2019-06
lang:   en
---

# I/O libraries

* How should HPC data be stored?
	- Large, complex, heterogeneous, esoteric, metadata ...
	- Parallel and random access
* Traditional relational databases poor fit
	- Cannot handle large objects
	- Many unnecessary features for HPC data
* MPI I/O is efficient but relatively low level

# I/O libraries

* I/O libraries can produce files with standardized format
	- Portable files that can be manipulated with external software
* Typically, I/O libraries support also metadata
	- Self-describing files
* Parallel I/O is typically build on top of MPI I/O
* **HDF5**, **NetCDF**, ADIOS, SIONlib

# HDF5

* A data model, library, and file format for storing and managing multidimensional data
* Can store complex data objects and meta-data
* File format and files are _portable_
* Possibility for parallel I/O on top of MPI-IO
* Library provides Fortran and C/C++ API
	- Third party interfaces for Python, R, Java
	- Many tools can work with HDF5 files (Paraview, Matlab, …)
* The HDF5 data model and library are complex

# HDF5 model
* Data can be organized hierarchically into groups in a tree and referred like in Unix filesystem
	- Data in root group: 				**/dataset1**
	- Data in the group group_1:	 **/group_1/dataset2**
* Root group is created automatically, other groups are created by the user
* Dataset stores multidimensional array of data elements
* Each object (group or dataset) can contain metadata
	- attributes = named values

# Investigating HDF5 files

* HDF5 installation includes command line tools `h5ls` and `h5dump` for investigating contents of HDF5 files

```commandline
$ h5ls example.hdf5 
AtomicDensityMatrices    Dataset {1, 11946}
AtomicNumbers            Dataset {158}
…
CartesianPositions       Dataset {158, 3}
Dimensions               Group ........
```
```commandline
$ h5dump -a Parameters/energyunit example.hdf5
…
DATA {
   (0): "Hartree« ....

```

# NetCDF

* NetCDF is a set of data formats, programming interfaces, and software libraries that help read and write scientific data files
* Originally developed for the Earth Sciences community
* File formats are self-describing and platform independent
* Optimized for access to subsets of large datasets
* Version 4 implemented on top of HDF5
* Also separate "Parallel netCDF" library

# NetCDF

* Data model is similar to HDF5
	- Data is stored in multidimensional datasets, which can be organized in a hierarchy
	- Metadata can be described in attributes
* Parallelization on top of MPI I/O

# Other libraries
* ADIOS - Adaptable IO System
	- An I/O abstraction framework	
	- Provides portable, fast,	 scalable, easy-­‐to-­‐use, metadata rich output
	- I/O  strategy  is  separated  from  the  user  code
* SIONlib
	- Simple replacement for posix I/O with task local I/O
	- No standard file format or hierarchy
	- Potentially good performance

# Summary
* I/O libraries are the recommended way for storing large data in production codes
* Portable files with standardized formats
	- No need to write application specific tools for investigating data
* Parallel I/O is normally built on top of MPI I/O
	-  MPI I/O concepts are utilized also in libraries
* Many scientific domains have standard conventions build on top of I/O libraries
