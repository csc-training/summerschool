#include <hdf5.h> // C-style HDF5 API
#include <mpi.h>

void hyperslab1D(const hid_t fileId) {

    // Create a 1D dataspace of length 10 and a corresponding dataset
    const hsize_t dims[1] = { 10 };
    hid_t dataspace = H5Screate_simple(1, dims, NULL);

    hid_t dataset = H5Dcreate(fileId, "MyDataset1D", H5T_NATIVE_INT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // Now each element in the dataset is 0 (default fill value in HDF5)


    // Create a *smaller* array that is to be written to the dataset
    const size_t dataLength = 8;
    int data[dataLength];
    // Give some nonzero values so that we can distinguish them from the default fill value
    for (size_t i = 0; i < dataLength; i++) {
        data[i] = (int) (i + 1);
    }

    // Define memory space (memspace) for writing, ie. shape of the source data
    const hsize_t memspaceDims[1] = { dataLength };
    hid_t memspace = H5Screate_simple(1, memspaceDims, NULL);

    // Perform a hyperslab selection so that the write will be done only to the first 'dataLength' elements of the dataspace

    const hsize_t offset[1] = { 0 };
    const hsize_t blockSize[1] = { dataLength };
    const hsize_t blockCount[1] = { 1 };

    // The following would also work (same offset): blockSize = { 1 }; blockCount = { 8 };

    herr_t status = H5Sselect_hyperslab(
        dataspace,          // Dataspace to operate on
        H5S_SELECT_SET,     // New selection
        offset,
        NULL,               // Default stride
        blockCount,
        blockSize
    );

    // Perform write with appropriate memspace/file space arguments
    status = H5Dwrite(
        dataset,
        H5T_NATIVE_INT,
        memspace,
        dataspace,
        H5P_DEFAULT,
        data
    );

    // Cleanup
    H5Dclose(dataset);
    H5Sclose(dataspace);
}

void hyperslab2D(const hid_t fileId) {

    // Create a 4D dataspace and a corresponding dataset
    const hsize_t dims[3] = { 6, 6 };
    hid_t dataspace = H5Screate_simple(2, dims, NULL);

    hid_t dataset = H5Dcreate(fileId, "MyDataset2D", H5T_NATIVE_INT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Create a 2x3 array, implemented as 1D for simplicity
    const size_t rows = 2;
    const size_t cols = 3;
    int data[rows * cols];

    // Give some nonzero values
    for (size_t i = 0; i < rows * cols; i++) {
        data[i] = (int) (i+1);
    }

    // Memspace: our logical array dimensions
    const hsize_t memspaceDims[2] = { rows, cols };
    hid_t memspace = H5Screate_simple(2, memspaceDims, NULL);

    // Top left hyperslab: select 1 block of 2x3 elements
    const hsize_t offset[2] = { 0, 0 };
    const hsize_t blockSize[2] = { 2, 3 };
    const hsize_t blockCount[2] = { 1, 1 };

    herr_t status = H5Sselect_hyperslab(
        dataspace,          // Dataspace to operate on
        H5S_SELECT_SET,     // New selection
        offset,
        NULL,               // Default stride
        blockSize,
        blockCount
    );

    // Write top left block
    status = H5Dwrite(
        dataset,
        H5T_NATIVE_INT,
        memspace,
        dataspace,
        H5P_DEFAULT,
        data
    );

    // Repeat with different selection: bottom right this time. We only need to change the offset
    const hsize_t offsetBottomRight[2] = {4, 3};
    status = H5Sselect_hyperslab(
        dataspace,
        H5S_SELECT_SET, // Discards the old selection
        offsetBottomRight,
        NULL,
        blockSize,
        blockCount
    );

    // Write bottom right block
    status = H5Dwrite(
        dataset,
        H5T_NATIVE_INT,
        memspace,
        dataspace,
        H5P_DEFAULT,
        data
    );

    // Cleanup
    H5Dclose(dataset);
    H5Sclose(dataspace);
}

int main(int argc, char** argv) {

    /* This program is serial and is not intended to be ran using MPI.
    Accidentally running with many MPI processes would lead to competing HDF5 writes and other mess, so to prevent this
    we still initialize MPI here and ensure only one rank does the work.
    */
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {

        // Create an HDF5 file, truncating existing contents if the file already exists
        hid_t fileId = H5Fcreate(
            "hyperslab_testing.h5",
            H5F_ACC_TRUNC,
            H5P_DEFAULT,
            H5P_DEFAULT
        );

        // ### Part 1 ###
        hyperslab1D(fileId);

        // ### Part 2 ###
        hyperslab2D(fileId);

        // Close the file
        H5Fclose(fileId);
    }

    MPI_Finalize();
    return 0;
}
