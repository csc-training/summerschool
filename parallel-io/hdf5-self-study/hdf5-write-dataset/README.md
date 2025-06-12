# hdf5-write-dataset

This exercise practices HDF5 file creation and writing of multidimensional datasets. Have a look at the unfinished code
at [`hdf5-dataset-write.cpp`](hdf5-dataset-write.cpp). This program creates a static array containing 16 integers which
are to be written as HDF5 datasets using different layouts. The main function calls some unfinished functions;
your task is to implement them as instructed below.

## Part 1

In the main function, open a new HDF5 file called `my_datasets.h5` using `H5Fcreate()`. You should instruct it to override
existing contents if the file exists already.

Then implement the following three functions:
- `writeDataset1D()` should write contents of input array to the input file as a 1D dataset.
- `writedataset2D()` should do the same, but write to a 4x4 2D dataset.
- `writedataset2D()` should do the same, but write to a 2x2x4 3D dataset.

In each case the functions should clean up after themselves by closing any HDF5 objects that they create. The HDF5 file
should be closed only in the main function.
- Use `H5Sclose()` and `H5Dclose()` for cleanup.

**Hint:** In each case you will be writing exactly 16 elements, only the logical layout will differ.
Therefore you can use `H5S_ALL` for the memory space and file space arguments in `H5Dwrite()`. Think about why this works.


## Part 2

Let's write some metadata attributes to the datasets. The main function already calls the `writeAttribute()` function,
which is currently empty. Implement it as follows:
- Given the file ID and dataset name as function inputs, open the dataset using `H5Dopen()`.
- Create a new `double`-valued attribute with name and value as specified by the function input. You will need a dataspace object for specifying shape of this attribute. Hint: NOT a "simple dataspace", those are for multidimensional arrays.
- Write the attribute to the dataset and clean up by closing any new HDF5 objects you created.

Inspect the resulting HDF5 file using `h5dump`. Can you identify the attributes from the output?
