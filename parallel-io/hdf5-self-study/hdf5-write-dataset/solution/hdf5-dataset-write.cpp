#include <array>

#include <hdf5.h> // C-style HDF5 API
#include <mpi.h>

void writeDataset1D(const std::array<int, 16>& arrayToWrite, const hid_t fileId, const char* datasetName) {

    // Specify dataspace shape
    const hsize_t dims[1] = { arrayToWrite.size() };

    // Create 1D dataspace
    hid_t dataspace = H5Screate_simple(1, dims, NULL);

    // Dataset creation
    hid_t dataset = H5Dcreate(
        fileId,           // Which file this dataset will reside in
        datasetName,      // Name of the dataset
        H5T_NATIVE_INT,   // Specify that the data consists of 'int' types
        dataspace,        // Dataspace that specifies shape of this dataset
        H5P_DEFAULT,      // Default link creation options
        H5P_DEFAULT,      // Default creation options
        H5P_DEFAULT       // Default access options
    );

    /* Write the dataset. From the docs:
        "H5Dwrite writes raw data from an application buffer to the specified dataset,
        converting from the datatype and dataspace of the dataset in memory to the datatype and dataspace of the
        dataset in the file. Specifying H5S_ALL for both the memory and file dataspaces indicates that the entire
        dataspace of the dataset is to be written to"
    */
    herr_t status = H5Dwrite(
        dataset,        // Dataset to write to
        H5T_NATIVE_INT, // Type of the data
        H5S_ALL,        // Memspace argument, specifies shape of the source data
        H5S_ALL,        // Filespace argument, describes where in the dataset we perform the write
        H5P_DEFAULT,    // Default data transfer options
        arrayToWrite.data()   // Pointer to source data
    );

    // Cleanup. But don't close the file as it was not opened by this function
    H5Dclose(dataset);
    H5Sclose(dataspace);
}

void writeDataset2D(const std::array<int, 16>& arrayToWrite, const hid_t fileId, const char* datasetName) {

    // This is analogous to the 1D case above, we only change the dataspace to be 4x4
    const hsize_t dims[2] = { 4, 4 };

    hid_t dataspace = H5Screate_simple(2, dims, NULL);

    // Can use H5P_DEFAULT for memspace / file space because although the dataspace and array shapes are different,
    // the number of elements are the same. HDF5 will access and write the data contiguously starting from the first array element
    hid_t dataset = H5Dcreate(fileId, datasetName, H5T_NATIVE_INT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    herr_t status = H5Dwrite(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, arrayToWrite.data());

    H5Dclose(dataset);
    H5Sclose(dataspace);
}

void writeDataset3D(const std::array<int, 16>& arrayToWrite, const hid_t fileId, const char* datasetName) {

    // This is analogous to the 1D and 2D cases, we only change the dataspace dimensions as instructed
    const hsize_t dims[3] = { 2, 2, 4 };

    hid_t dataspace = H5Screate_simple(3, dims, NULL);
    hid_t dataset = H5Dcreate(fileId, datasetName, H5T_NATIVE_INT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    herr_t status = H5Dwrite(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, arrayToWrite.data());

    H5Dclose(dataset);
    H5Sclose(dataspace);
}

void writeAttribute(const char* attributeName, double attributeValue, hid_t fileId, const char* datasetName) {

    // Open the dataset
    hid_t dataset = H5Dopen(fileId, datasetName, H5P_DEFAULT);
    // Error code will be < 0 if the operation failed (see the API specification)
    if (dataset < 0) {
        // Could not open the dataset, maybe it doesn't exist?
        printf("Error: could not open dataset [%s] for attribute writing\n", datasetName);
        return;
    }

    // Create dataspace for defining the layout of the attribute. Our metadata is a single number, so use scalar layout
    hid_t attributeSpaceId = H5Screate(H5S_SCALAR);

    // Create the attribute and associate it with our dataset
    hid_t attributeId = H5Acreate(
        dataset,                  // Handle to the dataset to which the attribute will be attached to
        attributeName,            // Name of the attribute
        H5T_NATIVE_DOUBLE,        // Datatype of the attribute
        attributeSpaceId,         // Handle to the Attribute Space
        H5P_DEFAULT,              // Default creation options
        H5P_DEFAULT               // Default access options
    );

    // Write the attribute to the attached dataset (skip error checking for simplicity)
    herr_t status = H5Awrite(attributeId, H5T_NATIVE_DOUBLE, &attributeValue);

    H5Aclose(attributeId);
    H5Sclose(attributeSpaceId);
    H5Dclose(dataset);
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

        // Create and initialize a 1D array of 36 integers
        std::array<int, 16> arrayToWrite;

        for (size_t i = 0; i < arrayToWrite.size(); i++) {
            arrayToWrite[i] = (int)i;
        }

        // Create a HDF5 file, truncating existing contents if the file already exists
        hid_t fileId = H5Fcreate(
            "my_datasets.h5",
            H5F_ACC_TRUNC,  // "truncate mode", ie. overwrite existing file. Read-write access is always implied
            H5P_DEFAULT,
            H5P_DEFAULT
        );

        writeDataset1D(arrayToWrite, fileId, "MyDataset1D");
        writeDataset2D(arrayToWrite, fileId, "MyDataset2D");
        writeDataset3D(arrayToWrite, fileId, "MyDataset3D");

        // Write some metadata to the datasets
        writeAttribute("CoolAttribute", 42.0, fileId, "MyDataset1D");
        writeAttribute("CoolestAttribute", 0.577, fileId, "MyDataset3D");

        // Close the file
        H5Fclose(fileId);
    }

    MPI_Finalize();
    return 0;
}
