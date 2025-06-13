#include <mpi.h>
#include "hdf5.h"

/* Example program for parallel HDF5 I/O.
Each MPI process writes their rank to a shared HDF5 file,
so that the resulting dataset is 1D and has 'ntasks' elements.
*/

int main(int argc, char** argv) {

    // Initialize MPI
    int rank, ntasks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    /*
    Configure HDF5 to use parallel I/O via MPI (by default HDF5 does not utilize MPI).
    The main way of configuring the behavior of HDF5 functions is via objects called Property Lists (plist).
    Property Lists are created using the `H5Pcreate` API call. When creating one, we must also specify a "category" or "type"
    that lets HDF5 know what the plist will be used for. Eg: does the plist configure file creation, file access, or something else.
    MPI-IO requires that multiple processes have access to the file, so we need a Property List that configures file access.
    To grant this access, we write info about our MPI setup (communicator) to the plist,
    then pass the plist to `H5Fcreate` instead of using default access settings.
    */

    hid_t plist = H5Pcreate(H5P_FILE_ACCESS);
    // Write our MPI communicator as a "property". Note the abbreviation: fapl = File Access Property List
    H5Pset_fapl_mpio(plist, MPI_COMM_WORLD, MPI_INFO_NULL);

    // Now create a new HDF5 file, passing our Property List as the "fapl" argument. This makes the created HDF5 file "MPI-aware".
    // Note that this is called from ALL processes in the MPI communicator (collective routine).
    hid_t file = H5Fcreate(
        "parallel_out.h5",  // file name
        H5F_ACC_TRUNC,      // Truncate mode, read/write access is implied
        H5P_DEFAULT,        // Default creation behavior
        plist               // Non-default File Access behavior to allow MPI-IO
    );

    // Create dataspace and dataset. Note that we wish to create only one dataset, NOT one dataset per process!
    // Hence all processes must call H5Dcreate() with the same dataspace (it is a collective routine).

    hsize_t dims[] = { (hsize_t) ntasks }; // Explicit cast from 'int' to HDF5 size type
    hid_t dataspace = H5Screate_simple(1, dims, NULL);
    // Dataset creation, call it "ranks". Default options are OK
    hid_t dataset = H5Dcreate(file, "ranks", H5T_NATIVE_INT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /*
    Setup file write so that writes from different processes don't overlap.
    We do this by having each rank select a different hyperslab from the dataspace.
    In this case we have 1D data and need to write one integer per rank,
    so our hyperslabs will also be 1D and have length of 1 (ie. block size 1, block count 1).
    */

    // Hyperslab start position in dataspace
    hsize_t start[] = { (hsize_t) rank }; // explicit cast to hsize_t

    // Number of blocks in the hyperslab
    hsize_t count[] = { 1 };

    // Hyperslab selection. This selection is "remembered" internally by the dataspace object. Skip error code check for brevity
    herr_t status = H5Sselect_hyperslab(
        dataspace,          // Dataspace to select from
        H5S_SELECT_SET,     // Selection operation code, can be used to modify existing selections. H5S_SELECT_SET just overrides any existing selection
        start,              // Hyperslab starting offset
        NULL,               // Stride (how many elements to move in each direction). NULL means 1
        count,              // How many blocks to select
        NULL                // Block size. NULL means 1
    );

    // Create a "memory space", ie. dataspace that specifies memory layout of the input data buffer. In our case the buffer is just 1 integer
    hid_t memspace = H5Screate_simple(1, count, NULL);

    // Each rank writes its own rank number to the dataset.
    // Individual non-collective write here, see the self-study notes for turning this into a collective write.
    H5Dwrite(
        dataset,
        H5T_NATIVE_INT,
        memspace,       // Memspace ID, specifies layout of input data
        dataspace,      // "File space": specifies where in the dataset we write the data. We can pass our dataspace here because it has an active hyperslab selection
        H5P_DEFAULT,    // Default data transfer properties (could configure for collective I/O)
        &rank           // Memory address of the data
    );

    // Close all handles and finalize. Note that these are collective operations
    H5Dclose(dataset);
    H5Sclose(dataspace);
    H5Sclose(memspace);
    H5Fclose(file);
    H5Pclose(plist);

    MPI_Finalize();
    return 0;
}
