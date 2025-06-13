#include <hdf5.h> // C-style HDF5 API
#include <mpi.h>

void partialWrite1D(const hid_t fileId) {

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

    // TODO: Write the 'data' array into the dataset. Ensure only the first 8 elements of the dataset are written to.

    // TODO: cleanup of HDF5 objects created in this function
}

void partialWrite2D(const hid_t fileId) {

    // Create a 2D dataspace and a corresponding dataset
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

    // TODO: write the 'data' array to the dataset as instructed in the exercise README.md.
    // The resulting dataset should have its top-left and top-right block replaced with 'data' contents


    // TODO: cleanup
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
            "partial_write.h5",
            H5F_ACC_TRUNC,
            H5P_DEFAULT,
            H5P_DEFAULT
        );

        // ### Part 1 ###
        partialWrite1D(fileId);

        // ### Part 2 ###
        partialWrite2D(fileId);

        // Close the file
        H5Fclose(fileId);
    }

    MPI_Finalize();
    return 0;
}
