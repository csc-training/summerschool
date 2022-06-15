---
title:  HDF5
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
- MPI-IO is efficient but relatively low level


# I/O libraries

- I/O libraries can produce files with standardized format
    - Portable files that can be manipulated with external software
- Typically, I/O libraries support also metadata
    - Self-describing files
- Parallel I/O is typically build on top of MPI-IO
- **HDF5**, **NetCDF**, ADIOS, SIONlib


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


# Key concepts

<small>

File
  : contiguous string of bytes (in memory or disc)

Group
  : collection of objects (including other groups)

Dataset
  : multidimensional array of data elements

Datatype
  : description of data element

Dataspace
  : description of dimensions of multidimensional array

Attribute
  : named value associated with group or dataset

Property list
  : parameters controlling options in library.
    `H5P_DEFAULT` refers to default options

</small>


# HDF5 library and programming model

- HDF5 library uses an object model
    - Implements *identifiers* to refer to objects
    - Identifiers are of type `hid_t` in C and `integer(kind=hid_t)`
      in Fortran
- Functions are grouped using prefixes, e.g. all dataset related functions
  start with H5D prefix
    - Fortran routines and constants have `_f suffix`


# HDF5 model

- Data can be organized hierarchically into groups in a tree and referred
  like in Unix filesystem
    - Data in root group: `/dataset1`
    - Data in the group *group_1*: `/group_1/dataset2`
- Root group is created automatically, other groups are created by the
  user
- Dataset stores multidimensional array of data elements
- Each object (group or dataset) can contain metadata
    - attributes = named values


# Investigating HDF5 files

- HDF5 installation includes command line tools *h5ls* and *h5dump*
  for investigating contents of HDF5 files

```shell
$ h5ls example.hdf5
AtomicDensityMatrices     Dataset {1, 11946}
AtomicNumbers             Dataset {158}
...
CartesianPositions        Dataset {158, 3}
Dimensions                Group
...

$ h5dump -a Parameters/energyunit example.hdf5
...
DATA {
   (0): "Hartree«
...
```


# Creating and opening an HDF5 file

- In order to open or create file user must define
    - File access mode
        - `H5F_ACC_EXCL` or `H5F_ACC_TRUNC` in create
        - `H5F_ACC_RDONLY` or `H5F_ACC_RDWR` in open
    - File creation property list (only in create)
    - File access property list
        - `H5P_DEFAULT` can be used for the default property list


# Creating / opening a file

- Creating a file

```c
#include <hdf5.h>

hid_t file_id; /* file identifier */

/* Create a new file using default properties. */
file_id = H5Fcreate("myfile.hdf5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
```

- Opening/closing a file

```c
/* Open a file using default properties. */
file_id = H5Fopen("myfile.hdf5", H5F_ACC_RDWR, H5P_DEFAULT, H5P_DEFAULT);

H5Fclose(file_id);
```


# HDF5 dataset

- A HDF5 dataset stores multidimensional array of data elements
- When creating a dataset, user must specify
    - Name of dataset
    - Data type
    - Data space
    - Storage properties
- Basic data types
    - `H5T_NATIVE_INT`, `H5T_NATIVE_FLOAT`, ...


# HDF5 dataspace

- An HDF5 dataspace is an object describing dimensionality and shape of an
  array
- Dataspace is required component of dataset definition
- Dataspaces are used to control data transfer during read or write
    - Layout of data in the source (memory during write, dataset during read)
    - Layout of data in the destination (dataset during write, memory
      during read)


# Creating a dataset

```c
int ndims = 2;
hsize_t dims[2] = { 3, 4 };
hid_t filespace_id, dset_id, plist_id;

// Dataspace definition is needed for dataset creation
filespace_id = H5Screate_simple(ndims, dims, NULL);

// Create a dataset of ints with given dataspace
dset_id = H5Dcreate(file_id, "board", H5T_NATIVE_INT, filespace_id,
                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
```


# Writing data

- Raw data in memory can have different type and/or size than in the HDF5
  file
- Therefore one has to specify
    - The dataset and its datatype and dataspace in memory
    - The dataspace of a dataset in the file
    - The dataset transfer property list
    - The data buffer
- When all these specifications have been made, the data can be written
  with function `H5Dwrite`


# Example: writing a file

```c
#include <hdf5.h>

int main(int argc, char *argv[]) {
    hid_t file_id, filespace_id, memoryspace_id, dset_id; // identifiers
    hsize_t dims[2] = {10,5};   // two dimensional 10x5 dataspace
    hsize_t *maxdims = NULL;    // fixed size dataspace

    int data[10][5];            // data to write

    /* ... Generate data ... */
```

# Example: writing a file (continued)

```c
    /* Create a file */
    file_id = H5Fcreate("test.hdf5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    /* Create a dataset */
    filespace_id = H5Screate_simple(2, dims, maxdims);
    dset_id = H5Dcreate(file_id, "set1", H5T_NATIVE_INT, filespace_id,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    /* Write data to file */
    memoryspace_id = H5Screate_simple(2, dims, maxdims);
    H5Dwrite(dset_id, H5T_NATIVE_INT, memoryspace_id, filespace_id,
             H5P_DEFAULT, data);

    /* Clean up */
    H5Sclose(filespace_id); H5Sclose(memoryspace_id); H5Dclose(dset_id); H5Fclose(file_id);
}
```


# Reading data

- Reading of data from an HDF5 file proceeds in opposite way to writing
    - Open the file
    - Open the dataset
    - Determine the dataspace of dataset
    - Determine dataspace and datatype in memory
    - Read the data from dataset to memory
- HDF5 provides a set of `H5xGet_yyyy` routines for obtaining information
  from datasets and dataspaces


# Data selection

- Dataspace can be used to describe just selected parts of data instead of
  the whole array, e.g.
    - Hyperslabs
    - Certain data points
- A hyperslab is defined by four arrays
    - offset starting point of the hyperslab
    - stride distance between successive elements
    - count number of blocks to select
    - block size of the block


# Data selection with hyperslab

![](img/hdf5-hyperslab.png)

```c
...
    hsize_t     offset[2] = {0, 1};
    hsize_t     stride[2] = {4, 1};
    hsize_t     count[2]  = {2, 1};
    hsize_t     block[2]  = {2, 1};

...
    dataspace_id = H5Dget_space (dataset_id);
    status = H5Sselect_hyperslab (dataspace_id, H5S_SELECT_SET, offset,
                                  stride, count, block);

    /* Data will be written only to the selection described by the hyperslab */
    status = H5Dwrite (dataset_id, H5T_NATIVE_INT, memspace_id,
                       dataspace_id, H5P_DEFAULT, sdata);
```




# Parallel I/O with HDF5

- So far we have used the default properties when opening the files
- For parallel I/O, special parameters are needed for file creation and
  dataset writing
    - HDF5 uses MPI-IO file routines for file operations
    - HDF5 library has to be compiled with parallel I/O support
- Each process can define different dataspace (i.e. hyperslab) for file
    - similarly to file view with MPI I/O


# Parallel IO with HDF5

- File access properties for parallel I/O

```c
plist_id = H5Pcreate(H5P_FILE_ACCESS);
H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
file_id = H5Fcreate("test.h5", H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
H5Pclose(plist_id);
```

- Dataset transfer property for parallel I/O


```c
plist_id = H5Pcreate(H5P_DATASET_XFER);
H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
H5Dwrite(dset_id, H5T_NATIVE_INT, memspace, filespace, plist_id, data);
H5Pclose(plist_id);
```

- Default write mode for datasets is individual



# High-level API

- For simple, serial pre- and postprocessing tasks one can use the
  high-level API of HDF5

```c
int ndims = 2;
int data[6] = {0,1,2,3,4,5};

hsize_t dims[2] = {2, 3};
hid_t file_id;

file_id = H5Fcreate("data.h5", H5F_ACC_TRUNK, H5P_DEFAULT, H5P_DEFAULT);
H5LTmake_dataset(file_id, "/data", 2, dims, H5T_NATIVE_INT, data);
```


# Summary

- HDF5 provides hierarchical data format for multidimensional array data
    - Files, datasets, dataspaces, selections
- Parallel IO can be performed by defining proper property lists for file
  creation/opening and dataset transfer
- Each process can work on part of the dataset
- HDF5 files are portable and can be analyzed also with several external
  programs
