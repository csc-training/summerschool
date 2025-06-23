#include <mpi.h>
#include <hdf5.h>

#include <vector>
#include <cassert>

int main(int argc, char** argv) {

    // Initialize MPI
    int rank, ntasks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    // Rank r will write r + 1 integers
    std::vector<int> data(rank + 1, rank);

    // Property List for using MPI
    hid_t plist = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist, MPI_COMM_WORLD, MPI_INFO_NULL);

    hid_t file = H5Fcreate(
        "output.h5",
        H5F_ACC_TRUNC,
        H5P_DEFAULT,        // Default creation behavior
        plist               // Non-default File Access behavior to allow MPI-IO
    );

    // Calculate dataspace 1D length
    size_t dataspaceSize = 0;
    for (size_t i = 0; i < ntasks; i++) {
        dataspaceSize += (i + 1);
    }

    // Create dataspace
    hsize_t dims[] = { (hsize_t) dataspaceSize };
    hid_t dataspace = H5Screate_simple(1, dims, NULL);
    // Create dataset
    hid_t dataset = H5Dcreate(file, "ranks", H5T_NATIVE_INT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Calculate hyperslab start for this rank
    hsize_t slabStart = 0;
    for (size_t i = 0; i < rank; i++) {
        slabStart += (i + 1);
    }

    const hsize_t start[] = { slabStart };
    // We can use block size of 1 => set number of blocks to match the number of elements this process should write
    const hsize_t count[] = { (hsize_t) (rank + 1) };

    herr_t status = H5Sselect_hyperslab(
        dataspace,
        H5S_SELECT_SET,
        start,
        NULL, // Default stride
        count,
        NULL // Default blocking
    );
    assert(status >= 0 && "Hyperslab selection error");

    // BONUS: Setup collective write
    hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
    status = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);

    hid_t memspace = H5Screate_simple(1, count, NULL);
    H5Dwrite(
        dataset,
        H5T_NATIVE_INT,
        memspace,
        dataspace,
        xfer_plist, // Bonus: collective write. Using H5P_DEFAULT here results in non-collective write
        data.data()
    );

    // Cleanup
    H5Sclose(memspace);
    H5Dclose(dataset);
    H5Sclose(dataspace);
    H5Fclose(file);
    H5Pclose(plist);
    MPI_Finalize();

    return 0;
}
