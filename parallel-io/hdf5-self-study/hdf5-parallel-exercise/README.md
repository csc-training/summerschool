# HDF5-parallel-exercise

This exercise practices the use of parallel I/O and hyperslabs in HDF5. You should be familiar with the [parallel HDF5 sample code](../hdf5-parallel-example/) before starting with this exercise.

Your task is to produce an HDF5 dataset where each MPI process writes a different amount of data. Consider the following setup:
- Each MPI process creates an array of `rank + 1` integers and initializes all array elements to `rank`. Ie: rank 0 has array `{ 0 }`, rank 1 has array `{ 1 1 }`, and so on.

Your task is to write the arrays to a HDF5 file as a continuous, ordered dataset using parallel I/O. Instructions:
1. Initialize MPI and setup a HDF5 file for parallel writing.
2. Initialize the data arrays as described above.
3. Create a one-dimensional HDF5 dataspace that is exactly as big as needed to hold the combined integers (`int` type) from all processes. Not too small, not too large.
4. Use hyperslabs and `H5Dwrite()` to write the arrays to file. The writes should be ordered by MPI rank.
5. Remember to cleanup any HDF5 objects that you created.

Examples of what the output of `h5dump` should look like on your file.
- With 2 MPI processes:
```c
HDF5 "output.h5" {
GROUP "/" {
   DATASET "ranks" {
      DATATYPE  H5T_STD_I32LE
      DATASPACE  SIMPLE { ( 3 ) / ( 3 ) }
      DATA {
      (0): 0, 1, 1
      }
   }
}
}
```
- With 5 MPI processes:
```c
HDF5 "output.h5" {
GROUP "/" {
   DATASET "ranks" {
      DATATYPE  H5T_STD_I32LE
      DATASPACE  SIMPLE { ( 15 ) / ( 15 ) }
      DATA {
      (0): 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4
      }
   }
}
}
```

You may use the [parallel HDF5 sample code](../hdf5-parallel-example/) as a starting point or start from scratch.

**Bonus:** Modify your program so that the parallel write operation is collective.
