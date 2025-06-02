#include <vector>

#include <hdf5.h> // C-style HDF5 API
#include <mpi.h>

/* HDF5 starter example. This creates a 1D array and writes it as a 2D dataset to a new HDF5 file.
We also write a double-valued metadata field.
*/

int main(int argc, char **argv) {

    // Initialize MPI
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // This example program is serial, ie. we let only rank 0 do stuff
    if (rank == 0) {

        // Create an array of integers. This is 1D, but we will interpret and write it as a 2D matrix by specifying row/column counts.
        // This is a common way of implementing dynamical multidimensional arrays of arbitrary shapes.
        const size_t rows = 4;
        const size_t columns = 5;

        std::vector<int> matrix(rows * columns);
        for (size_t i = 0; i < rows*columns; i++)
        {
            matrix[i] = (int) i;
        }

        // Now create a new HDF5 file.
        // In HDF5 the file creation function is `H5Fcreate`, which returns an integer ID to the created object ("handle to object").
        // The last 2 arguments can be used for advanced configurations; here we just use the defaults by passing `H5P_DEFAULT`.
        hid_t fileId = H5Fcreate(
            "matrix.h5",    // file name
            H5F_ACC_TRUNC,  // "truncate mode", ie. overwrite existing file. Read-write access is always implied
            H5P_DEFAULT,    // Default file creation options
            H5P_DEFAULT     // Default file access options (we explore this in a follow-up exercise)
        );

        // Next, create HDF5 dataspace to define dataset shape. We want it to be a 2D array.

        // Shape of the dataspace
        hsize_t dims[2] = {rows, columns};

        // 2D dataspace creation. "simple dataspace" is HDF5 jargon for a multidimensional array
        hid_t dataspaceId = H5Screate_simple(
            2,      // 2D dataspace
            dims,   // Shape, ie. how many rows and columns
            NULL    // Could limit maximum row/column count here. NULL means unlimited
        );

        // Create the dataset, no actual I/O here yet.
        hid_t datasetId = H5Dcreate(
            fileId,           // Which file this dataset will reside in
            "VeryCoolMatrix", // Name of the dataset
            H5T_NATIVE_INT,   // Specify that the data consists of 'int' types. This way HDF5 knows how to interpret the data on any platform (portability!)
            dataspaceId,      // Dataspace to use for this dataset, ie. data shape.
            H5P_DEFAULT,      // Default link creation options. Advanced feature: "links" in HDF5 behave like symlinks in UNIX
            H5P_DEFAULT,      // Default creation options
            H5P_DEFAULT       // Default access options
        );

        // Perform the actual write. We pass the `matrix.data()` pointer to our data, and use the memspace argument to specify
        // how the full data should be obtained starting from this memory address.
        // In this case our input data is contiguous and fits nicely in the dataset, so we can use `H5S_ALL` to let HDF5 know
        // it can use the dataset shape to access the data. Same for the filespace argument: it can write the entire dataset.
        // Return value is an error code and will be < 0 if there was an error.
        // We skip manual error checking here for simplicity (HDF5 also has a built-in validation layer).
        herr_t status = H5Dwrite(
            datasetId,      // Dataset to write to
            H5T_NATIVE_INT, // Type of the data, should match the type used when defining the dataset
            H5S_ALL,        // Dataspace describing layout of the memory buffer
            H5S_ALL,        // Dataspace describing where in the dataset we write to
            H5P_DEFAULT,    // Default data transfer options
            matrix.data()   // Pointer to data
        );

        // Write some metadata using HDF5 attributes. Here we just write a dummy floating point number.
        // In a real program this metadata could represent eg. value of an input parameter used when producing the data.
        const double dummyMetadata = 42.0;

        // Create dataspace for defining the layout of the attribute. Our metadata is a single number, so use scalar layout
        hid_t attributeSpaceId = H5Screate(H5S_SCALAR);

        // Create the attribute and associate it with our dataset
        hid_t attributeId = H5Acreate(
            datasetId,                // Handle to the dataset to which the attribute will be attached to
            "DummyAttribute",         // Name of the attribute
            H5T_NATIVE_DOUBLE,        // Datatype of the attribute
            attributeSpaceId,         // Handle to the Attribute Space
            H5P_DEFAULT,              // Default creation options
            H5P_DEFAULT               // Default access options
        );

        // Write the attribute to the attached dataset (skip error checking for simplicity)
        status = H5Awrite(attributeId, H5T_NATIVE_DOUBLE, &dummyMetadata);

        // Cleanup by closing (deallocating) all HDF5 objects that we created. Do this in reverse order to be on the safe side
        H5Aclose(attributeId);
        H5Sclose(attributeSpaceId);
        H5Dclose(datasetId);
        H5Sclose(dataspaceId);
        H5Fclose(fileId);
    }

    MPI_Finalize();
    return 0;
}
