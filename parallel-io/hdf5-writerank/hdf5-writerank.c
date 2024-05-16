#include "hdf5.h"
#include <mpi.h>

int main(int argc, char **argv) {
    
    // Initialize MPI
    int myproc, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    // Create a new property list for file access
    hid_t plist = H5Pcreate(H5P_FILE_ACCESS);
    // Store MPI IO communicator info to the file access property list
    H5Pset_fapl_mpio(plist, MPI_COMM_WORLD, MPI_INFO_NULL);
    // Create a new HDF5 file named "parallel_out.h5"
    hid_t file = H5Fcreate("parallel_out.h5", H5F_ACC_TRUNC, H5P_DEFAULT, plist);
    // Create a new simple dataspace for the file and open for access
    hid_t dataspace = H5Screate_simple(1, (const hsize_t[]){numprocs}, NULL);
    // Create a new dataset named "MPI_RANKS" for 'file'
    hid_t dataset = H5Dcreate(file, "MPI_RANKS", H5T_NATIVE_INT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // Number of blocks to be included in the hyperslab region
    hsize_t count[] = {1};
    // Select a hyperslab region of the file dataspace
    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, (const hsize_t[]){myproc}, NULL, count, NULL);
    // Create a new simple dataspace for the memory buffer and open for access
    hid_t memspace = H5Screate_simple(1, count, NULL);
    // Each rank writes its own rank number (partially constructing 'dataset') into a file
    H5Dwrite(dataset, H5T_NATIVE_INT, memspace, dataspace, H5P_DEFAULT, &myproc);

    // Close all handles and return
    H5Dclose(dataset);
    H5Sclose(dataspace);
    H5Sclose(memspace);
    H5Fclose(file);
    H5Pclose(plist);
    MPI_Finalize();
    return 0;
}
