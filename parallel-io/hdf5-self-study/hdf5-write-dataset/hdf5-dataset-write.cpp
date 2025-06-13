#include <array>

#include <hdf5.h> // C-style HDF5 API
#include <mpi.h>

void writeDataset1D(const std::array<int, 16>& arrayToWrite, const hid_t fileId, const char* datasetName) {

    // TODO: write the input array to the input file as a new 1D dataset.
    // Assume 'fileId' is already open.
    // Remember to close any HDF5 resources you open here before exiting this function, but do not close the file itself

}

void writeDataset2D(const std::array<int, 16>& arrayToWrite, const hid_t fileId, const char* datasetName) {

    // TODO: like writeDataset1D, but write the array as a 4x4 dataset
}

void writeDataset3D(const std::array<int, 16>& arrayToWrite, const hid_t fileId, const char* datasetName) {

    // TODO: like writeDataset1D, but write the array as a 2x2x4 dataset
}


// For part 2 of the exercise
void writeAttribute(const char* attributeName, double attributeValue, hid_t fileId, const char* datasetName) {

    // TODO: write a 'double' attribute of given name and value to the specified file and dataset
    // Remember to close any HDF5 resources you open here
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

        // Create and initialize a 1D array of 16 integers
        std::array<int, 16> arrayToWrite;

        for (size_t i = 0; i < arrayToWrite.size(); i++) {
            arrayToWrite[i] = (int)i;
        }

        // ### PART 1 ###

        // TODO: open a new HDF5 file, call it 'my_datasets.h5'.
        // Override the file if it exists already (hint: the HDF5 file creation function has a flag for this)


        hid_t fileId = -1; // replace with a proper file open call

        // Write some datasets to the file. TODO: implement these functions
        writeDataset1D(arrayToWrite, fileId, "MyDataset1D");
        writeDataset2D(arrayToWrite, fileId, "MyDataset2D");
        writeDataset3D(arrayToWrite, fileId, "MyDataset3D");

        // ### PART 2 ###

        // Write some metadata to the datasets
        writeAttribute("CoolAttribute", 42.0, fileId, "MyDataset1D");
        writeAttribute("CoolestAttribute", 0.577, fileId, "MyDataset3D");

        // TODO: close the file
    }

    MPI_Finalize();
    return 0;
}
