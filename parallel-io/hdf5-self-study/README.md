# Self study module on HDF5

HDF5 is a file format that has become popular in scientific and industrial computing due to its flexibility,
I/O performance and portability. HDF5 files are designed for storing large multidimensional arrays in a compact binary
format and work well in parallel filesystems such as Lustre, making HDF5 a good fit for many I/O tasks in HPC.

These notes give an introduction to the HDF5 file format and its C/Fortran API.
"API" (Application Programming Interface) refers to the set of functions and objects exposed by the HDF5 programming
library that we can use from our code.

We will cover the following:
- Creating and writing HDF5 datasets into a file, and reading them
- Writing user-specified metadata for the datasets using HDF5 attributes
- Investigating HDF5 file contents with command line tools
- Using MPI parallel I/O with HDF5 files
- Using HDF5 hyperslabs to selectively operate on parts of a dataset

You can find the official HDF5 documentation [here](https://support.hdfgroup.org/documentation/hdf5/latest/index.html).
Especially useful are the User Guide and Reference Manual tabs.

**TODO:** Prepare Fortran example codes!

## The HDF5 programming interface (= API)

The HDF5 API provides functions for creating and manipulating HDF5 files and datasets within them.
The API is very flexible, giving the programmer full control over how datasets should be created or accessed.
The price to pay for this flexibility is that the programming interface is rather verbose and abstract.
For example, many API calls require the programmer to configure their behavior by passing **HDF5 Property List** objects
as function argument, but in many cases the default behavior is sufficient in which case we instead pass `H5P_DEFAULT`.

Throughout these notes we will use the "standard" C-style API, accessible in C/C++ by including the header `hdf5.h`.
Most API functions that create HDF5 objects (eg. file or dataset creation) return an integer identifier of type `hid_t`
to the created resource, instead of returning a direct pointer to it. Likewise, functions that operate on HDF5 objects
take in these IDs, or **handles**, as arguments. This is a somewhat common way of hiding implementation details of
library objects or structs from the programmer. Some routines return an error code (integer of `herr_t` type) that can
be used for manual error checking. Error checking is not always necessary because many HDF5 routines have a built-in
validation layer that complains about incorrect use of the API.

In addition to the C-style API used here, you should be aware of the following alternatives when implementing HDF5 in
your own codes:
- [Official "high-level" HDF5 APIs](https://docs.hdfgroup.org/archive/support/HDF5/doc/HL/index.html).
These are simplified APIs that are considerably less verbose than the "full" API. In practice they are wrappers around
the full API and aim to streamline common operations such as dataset read/write. The main downside is that some advanced
features such as parallel I/O are not available.
- For C++ users, the header `H5Cpp.h` provides C++ style bindings to the full API.
Concepts from the C-style API carry over. It is of course valid to use the C-API also in C++ if you prefer.
- The Python package [`h5py`](https://pypi.org/project/h5py/) has both a high-level API for common HDF5 tasks, and also
a Python wrapper around the low-level C-API.

### HDF5 Fortran API

The [Fortran API](https://docs.hdfgroup.org/archive/support/HDF5/doc/fortran/index.html) is rather similar to the
C-style API, but has the following differences:
- Function names in Fortran are suffixed by `_f`. Eg: `H5Dcreate` in C becomes `h5dcreate_f` in Fortran.
- Some functions have the error code as an additional output argument
- Order of arguments may sometimes vary relative to the C version. Input arguments come first, then output parameters
(including the error code), then optional input parameters.


## HDF5 file structure

HDF5 files are binary files intended for storing arbitrary N-dimensional datasets, where each element in the dataset may
itself be a complex object ("heterogeneous data").
There is no limit on how big the datasets can be; HDF5 can hold arbitrarily large amounts of data.
HDF5 has a complex, filesystem-like structure that allows one file to hold many datasets in an organized fashion.

![](./img/hdf5_structure.png)
*HDF5 file structure. "Groups" are analogous to directories on a Unix-like filesystem, and "datasets" then correspond to files.*


The HDF5 data model separates the **shape** of data from the dataset itself. Data shape (number of rows, columns etc.)
in HDF5 is called a **dataspace**. Dataspaces and datasets must be managed separately by the programmer, and creation of
a dataset requires a valid dataspace.
- Analogy from Python: `numpy` arrays and their `numpy.shape` objects.

![](./img/hdf5_dataset.png)
*Example HDF5 dataset and its metadata. Image taken from https://portal.hdfgroup.org/documentation/hdf5/latest/_intro_h_d_f5.html.*

The minimal steps for creating an HDF5 file and writing a dataset to it are as follows:
1. Create the file using [`H5Fcreate()`](https://docs.hdfgroup.org/archive/support/HDF5/doc/RM/RM_H5F.html#File-Create),
with appropriate creation flags and configuration options.
2. Create a [**dataspace**](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)
to represent shape of the data. Usually we are interested in writing N-dimensional arrays; dataspaces corresponding to
these are called "simple" in HDF5. A simple dataspace can be created with [`H5Screate_simple()`](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)
in which we specify the dimensionality and number of elements along each dimension.
3. Create a [**dataset**](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d__u_g.html) by calling
[`H5Dcreate()`](https://docs.hdfgroup.org/archive/support/HDF5/doc/RM/RM_H5D.html#Dataset-Create). In this function call
we specify which file this dataset is to be created in, type of data that we are storing (eg. integers of floats), and a
valid dataspace for defining dataspace shape.
4. Call [`H5Dwrite()`](https://docs.hdfgroup.org/archive/support/HDF5/doc/RM/RM_H5D.html#Dataset-Write) to write data to
into the dataset. We have to specify the target dataspace, type of data to be written and a valid pointer to memory
address where the source data resides. We must also pass _two_ dataspaces for specifying shapes of source and target
data:
    - A "memory space" or **memspace**, which defines how the source data is aligned in memory. This is like specifying
    the number of elements to get starting from the input memory address in "standard" I/O routines. If the source data
    is contiguous and has same logical shape as the dataspace used by the dataset, we may use the special keyword
    `H5S_ALL`; otherwise a valid memspace must be manually created (via `H5Screate_simple()`) and passed to `H5Dwrite()`.
    - A "file space", which is an another dataspace object specifying where in the dataset the data should be written to.
    Passing `H5S_ALL` means we write the full memspace.
A more detailed specification of memspace and file space semantics can be found in the documentation linked above.

This is a lot of programming overhead just for outputting data to a file! For simple writes most of this machinery is
indeed unnecessarily complicated, but becomes very useful when working with complex or parallel data.


### Writing metadata via HDF5 attributes

HDF5 [**attributes**](https://portal.hdfgroup.org/documentation/hdf5/latest/_h5_a__u_g.html) are a special data
structure intended for storing metadata. Metadata *could* be stored as standard HDF5 datasets, however this can be
inefficient because metadata is usually small compared the actual data. HDF5 attributes are similar to datasets, but
optimized for small metadata that can be *attached* to datasets.

Attributes can be created using the [`H5Acreate function`](https://docs.hdfgroup.org/archive/support/HDF5/doc/RM/RM_H5A.html#Annot-Create).
This requires the following arguments:
- A valid dataset ID to which the attribute will be attached to
- A name for the attribute (string)
- Type of the attribute (built-in HDF5 type identifier)
- Dataspace ID that specifies **shape** of the metadata. For example, a simple scalar-valued metadata field should use
dataspace created with the `H5S_SCALAR` flag.
- Creation/access configuration options (`H5P_DEFAULT` gives default behavior)

Once created, the attribute can be written to file with [`H5Awrite`](https://docs.hdfgroup.org/archive/support/HDF5/doc/RM/RM_H5A.html#Annot-Write).
The syntax is considerably simpler than the write function for datasets.


### Reading HDF5 files

So far we have only discussed writing to HDF5 files. The API for file reading is rather similar, but there is no need to
create dataspaces since dataset shapes can be inferred from the file.

Below is an example read of a 2D dataset using the C-API.
```c
// 2D array for holding the read data
int data[rows][columns];

// Open HDF5 file
hid_t fileId = H5Fopen(
    "someFile.h5",  // File name
    H5F_ACC_RDONLY, // Read-only access
    H5P_DEFAULT     // Default access properties
);

// Open a dataset in the file
datasetId = H5Dopen(
    fileId,         // File to read from
    "someDataset",  // Name of the dataset
    H5P_DEFAULT     // Default access properties
);

H5Dread(
    datasetId,      // Dataset to read from
    H5T_NATIVE_INT, // Type of data to read, here `int` type
    H5S_ALL,        // Memspace
    H5S_ALL,        // File spac.  H5S_ALL for this and memspace means we read the full dataset
    H5P_DEFAULT,    // Default transfer properties
    data            // Pointer to the array to which the data will be stored
);

// Cleanup
H5Dclose(datasetId);
H5Fclose(fileId);
```
If the types/names/shapes of stored data are unknown, we should query them from the file or dataset using the API.
See eg. [`H5Dget_type`](https://docs.hdfgroup.org/archive/support/HDF5/doc/RM/RM_H5D.html#Dataset-GetType).

Often it it convenient to inspect HDF5 file contents directly from the command line. For this the commands `h5ls` and
`h5dump` can be used. These tools are usually bundled with HDF5 installations, or on computing clusters become available
after loading an appropriate HDF5 module. We practice using these tools in the case study below.


### Case study: Writing a 2D dataset

Read through the example code (C++ or Fortran) in [`hdf5-write-matrix`](./hdf5-write-matrix/). This program creates a
contiguous 1D array and writes it to an HDF5 file as a 2D dataset (a common way of implementing multidimensional arrays
is to use a large 1D array and simply interpret it as N-dimensional). A metadata field is also written using a `double`
attribute.

Exercises:
1. Compile and run the program (without MPI, or just 1 MPI process). It should produce a file called `matrix.h5` in the
working directory. See [](hdf5-exercise-instructions.md) for compilation instructions.
2. Use the HDF5 command line tools to inspect contents of the file.
```bash
h5ls matrix.h5
h5dump matrix.h5
```
`h5ls` gives a list of datasets in the file and their shapes. Ensure you understand the output. `h5dump` gives a full
dump of the file contents. Can you identify the matrix values, and the metadata?

3. How would you modify the example code to instead produce a 3D dataset?


## Parallel write with HDF5 and MPI

The HDF5 development library can be compiled with MPI support to allow many MPI processes to operate on shared HDF5
files. In the API, parallel access to HDF5 files is configured at the time of file creation or opening (`H5Fcreate()` or
`H5Fopen()`) using the "file access property" argument to the function call. So far we have bypassed this argument by
setting it to `H5P_DEFAULT`; in many cases the default behavior is indeed sufficient, but for parallel I/O we must
configure the file access manually.

The HDF5 API uses objects called [**Property Lists**](https://portal.hdfgroup.org/documentation/hdf5/latest/_h5_p__u_g.html)
for configuring API calls. These are collections of configurable **properties** specified by the HDF5 standard.
Property Lists come in different flavors, or **Property List Classes**, depending on what properties they manage.
- *Side note*: Notice how similar this design is to Object Oriented Programming (OOP), despite OOP not being a built-in
feature in neither C nor Fortran.

In the case of parallel I/O, we need a Property List for configuring file access specifically.
We can create one as follows:
```c
hid_t plist = H5Pcreate(H5P_FILE_ACCESS);
```
Next we tell the Property List about our MPI setup, on in HDF5 language, we must set the relevant **property**.
HDF5 provides a [function specifically for this purpose](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1):
```c
// Note the abbreviation: fapl = File Access Property List
herr_t status = H5Pset_fapl_mpio(plist, MPI_COMM_WORLD, MPI_INFO_NULL);
```
The last argument is of `MPI_Info` type and could be used for advanced MPI-IO configuration.
The return value is an error code (negative value means failure).

With this, the following call creates a new HDF5 file and opens it for parallel access in all MPI ranks:
```c
hid_t file = H5Fcreate(
    "parallelFile.h5",  // file name
    H5F_ACC_TRUNC,      // Truncate if file exists. Read/write access is implied
    H5P_DEFAULT,        // Default creation behavior
    plist               // Non-default File Access behavior to allow MPI-IO
);
```
This should be called from all MPI processes: it is a *collective* operation. Same for `H5Fclose()` when cleaning up.

### Using hyperslabs to avoid overlapping writes

How do we write data to the file while ensuring that parallel writes from different processes do not mess with each
other? Recall that in MPI-IO we could calculate a different `MPI_Offset` for each rank and pass this to I/O operations
to read/write different sections of the file stream.

HDF5 uses a more general (and abstract) way of specifying offsets: **hyperslabs**. More specifically, hyperslabs are
used to *select* subregions of dataspaces for data manipulation or I/O, hence the name: they are slices of N-dimensional
spaces. Hyperslabs can be useful also in serial applications that need to operate only on specific parts of a dataset.
Here we demonstrate their use with parallel dataset writes.

Hyperslab selection is organized in terms of **blocks** of dataspace elements.Eg: for a 2D dataspace, block size of
`(2, 2)` means we would select one or more subspaces, each containing 2x2 elements. Block size of `(1, 1)` would mean
we'd just select individual elements (default behavior). We can select hyperslabs (one or more) from a dataspace using
[H5Sselect_hyperslab()](https://docs.hdfgroup.org/archive/support/HDF5/doc/RM/RM_H5S.html#Dataspace-SelectHyperslab).
It takes in the following arguments:
- Dataspace ID
- A "selection operation code", ie. what kind of selection are we performing. For example, `H5S_SELECT_SET` will replace
any existing selection with the new selection, `H5S_SELECT_OR` will add any new hyperslabs to an existing selection,
and so on.
- The following 4 `hsize_t` arrays, length of each one must match dataspace dimensionality/rank:
    - Starting offset: How many elements to skip in each direction before starting selection.
    - Stride: Specifies how the dataspace is traversed when selecting elements. `stride[i]` is the number of elements to
    move in direction `i`, ie. elements to be selected are `offset[i]`, `offset[i] + stride[i]`, `offset[i] + 2*stride[i]`
    etc. Passing `NULL` stride defaults to 1 in all directions, meaning a contiguous selection.
    - Block count: How many blocks to select in each direction.
    - Block size: How many elements to include in one block, as discussed above. `NULL` means 1 in each direction
    (single-element blocks).

See the following figure for a demonstration of hyperslab selection. More hyperslab visualizations can be found on the
[HDF5 homepage](https://portal.hdfgroup.org/documentation/hdf5/latest/_l_b_dset_sub_r_w.html).

![](./img/hdf5-hyperslabs.svg)
*Hyperslab selection example*

HDF5 "remembers" which hyperslab of the dataspace is currently selected and allows dataspace operations only in the
active selection. This is useful for parallel I/O: each MPI process can select a unique hyperslab based on its MPI rank,
and use `H5Dwrite/H5Dread` to perform I/O only in its own hyperslab.


### Collective I/O HDF5

Recall that collective operations must be called from all MPI processes (for example, `MPI_Bcast()` or
`MPI_File_write_at_all()`). Collective operations give the MPI implementation more opportunities for optimizations.

In parallel HDF5, some of the API calls can be made collective with Property List configurations, and some are defined
to always be collective. For example, dataset creation with `H5Dcreate()` is always collective, whereas dataset write
with `H5Dwrite()` is non-collective by default. You can read more about collectiveness requirements of the HDF5 API
calls in [the docs](https://support.hdfgroup.org/documentation/hdf5/latest/collective_calls.html).

Dataset read and write operations can be made collective by configuring their Transfer Property List argument
accordingly. As before with parallel file access, we create a new Property List and use a dedicated property setter call:
```c
// Common abbreviation: xfer_plist = Transfer Property List
hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
herr_t status = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
// Pass this to H5Dwrite()/H5Dread(). Example:
H5Dwrite(dataset, H5T_NATIVE_INT, memspace, filespace, xfer_plist, data_pointer);
```

### Case study: parallel write

Read through the example code [`hdf5-parallel-example`](./hdf5-parallel-example/).
The program demonstrates parallel HDF5 I/O by writing one integer from each MPI process.

**Tasks:**
1. Ensure you understand the steps for creating and opening an HDF5 file for parallel access.
2. Compile and run the program with 4 MPI processes. Inspect the output file `parallel_out.h5` using `h5dump`.
Try to understand the relevant hyperslab selection logic used in the example code.
3. Run the program again with a different number of MPI processes and verify that the dataset shape and contents have
changed.

**Exercise:**

Write a modified version of the program as instructed in [`hdf5-parallel-exercise`](./hdf5-parallel-exercise/README.md).
