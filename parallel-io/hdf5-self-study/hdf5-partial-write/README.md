# hdf5-partial-write

This exercise practices the use of HDF5 hyperslabs for writing partial data to a larger dataset. This is a common task
when working with existing datasets, and especially in programs that do parallel I/O.

Starting point for this exercise is this unfinished code: [`hdf5-partial-write.cpp`](hdf5-partial-write.cpp). The main
function creates a new HDF5 file an then calls two unfinished functions that should write datasets into it. Your task is
to implement these as instructed below.

## Part 1

Have a look at the function `partialWrite1D()`. It creates a 1D dataset of 10 elements, and a smaller integer array of 8
elements. Implement the function so that it writes only the 8 elements to the dataset, leaving leftover dataset elements
untouched.

First, try performing the write as before, using `H5S_ALL` in the memspace and field space fields and inspect the resulting file with `h5dump`.
You should see that the last two values are gibberish. This is because with `H5S_ALL` HDF5 ends up doing out-of-bounds
access to the source array and gets garbage data for the last elements.

Fix this by using appropriate dataspaces as the memspace and file space arguments:
- Create a dataspace for describing shape of the source array (ie. the memspace)
- Perform a hyperslab selection on the original dataspace to select only as many elements that you are writing
- Use `H5Dwrite` with appropriate arguments to perform the write.

Verify with `h5dump` that only sensible data was written. Example output:
```c
DATASET "MyDataset1D" {
    DATATYPE  H5T_STD_I32LE
    DATASPACE  SIMPLE { ( 10 ) / ( 10 ) }
    DATA {
    (0): 1, 2, 3, 4, 5, 6, 7, 8, 0, 0
    }
}
```

What happens if you try using a memspace that is smaller or larger than the active hyperslab selection?

## Part 2

Next let's work on the function `partialWrite2D()`. This creates a 6x6 2D dataset and a 2x3 array. IMplement rest of the
function so that the array is written both to the top-left AND bottom-right blocks of the dataset.

Example output of `h5dump`:
```c
DATASET "MyDataset2D" {
    DATATYPE  H5T_STD_I32LE
    DATASPACE  SIMPLE { ( 6, 6 ) / ( 6, 6 ) }
    DATA {
    (0,0): 1, 2, 3, 0, 0, 0,
    (1,0): 4, 5, 6, 0, 0, 0,
    (2,0): 0, 0, 0, 0, 0, 0,
    (3,0): 0, 0, 0, 0, 0, 0,
    (4,0): 0, 0, 0, 1, 2, 3,
    (5,0): 0, 0, 0, 4, 5, 6
    }
}
```

**Hint:** How many write operations do you need?
