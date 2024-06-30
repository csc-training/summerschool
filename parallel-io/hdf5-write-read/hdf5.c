#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <hdf5.h>
#include <mpi.h>
#include <assert.h>

#define DATASIZE   64

void h5_writer(int, int *, int);
void h5_reader(int, int *, int);

// On LUMI, remember to module load cray-hdf5-parallel/1.12.2.7
// Compile with e.g. cc hdf5.c -lhdf5 -o hdf5_write_read.exe

Does not work!!!!!

int main(int argc, char *argv[])
{
    int my_id, ntasks, i, localsize;
    int *writevector, *readvector;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

    if (ntasks > 64) {
        fprintf(stderr, "Datasize (64) should be divisible by number "
                "of tasks.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    if (DATASIZE % ntasks != 0) {
        fprintf(stderr, "Datasize (64) should be divisible by number "
                "of tasks.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    localsize = DATASIZE / ntasks;
    writevector = (int *) malloc(localsize * sizeof(int));
    readvector = (int *) malloc(localsize * sizeof(int));
    for (i = 0; i < localsize; i++)
        writevector[i] = i + 1 + localsize * my_id;

    h5_writer(my_id, writevector, localsize);

    h5_reader(my_id, readvector, localsize);

    for (i = 0; i < localsize; i++) 
	 assert(writevector[i]==readvector[i]);

    free(writevector);
    free(readvector);
    MPI_Finalize();
    return 0;
}

void h5_writer(int my_id, int *localvector, int localsize)
{
    herr_t status;
    hid_t plist_id, dset_id, filespace, memspace, file_id;
    hsize_t dims, counts, offsets;

    /* Create the handle for parallel file access property list
       and create a new file */
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    // Store MPI IO communicator info to the file access property list
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

    // Create a new HDF5 file named "data.h5"
    file_id = H5Fcreate("data.h5", H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Pclose(plist_id);

    /* Create the dataset */
    // Create a new simple dataspace for the file and open for access
    const hsize_t dim_arr[] = {DATASIZE};  // one-dimensional array of size rank specifying the size of each dimension of the dataset.
    const hsize_t *maxdims = NULL;
    filespace = H5Screate_simple(1, dim_arr, maxdims);  // Create a one-dimensional dataspace with a specific size (given by DATASIZE) and no maximum dimensions (due to NULL).

    dset_id = H5Dcreate(file_id, "data", H5T_NATIVE_INT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);  // Create a new dataset named "data" for 'file_id'.

    /* Select a hyperslab of the file dataspace */

    // Select a hyperslab region of the file dataspace
    //H5Sselect_hyperslab(filespace, H5S_SELECT_SET, (const hsize_t[]){(hsize_t)myproc}, NULL, count, NULL);

    counts = localsize;  // Number of blocks to be included in the hyperslab region
    offsets = my_id * localsize;
    const hsize_t *stride = NULL;
    const hsize_t *block = NULL;
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, &offsets, &stride, &counts, block);  // Modifies the selection in the filespace by defining a hyperslab based on offsets and counts (no stride, i.e. contiguous selection).

    /* Now we can write our local data to the correct position in the
       dataset. Here we use collective write, but independent writes are
       also possible. */
    memspace = H5Screate_simple(1, &counts, NULL);  // Create a new simple dataspace for the memory buffer and open for access
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    status = H5Dwrite(dset_id, H5T_NATIVE_INT, memspace, filespace, plist_id, localvector);

    /* Close all opened HDF5 handles */
    H5Dclose(dset_id);
    H5Pclose(plist_id);
    H5Sclose(filespace);
    H5Sclose(memspace);
    H5Fclose(file_id);
}


void h5_reader(int my_id, int *localvector, int localsize)
{
    herr_t status;
    hid_t plist_id, dset_id, filespace, memspace, file_id;
    hsize_t dims, counts, offsets;

    /* Create the handle for parallel file access property list
       and open a file for reading */
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
    file_id = H5Fopen("data.h5", H5F_ACC_RDONLY, plist_id);
    H5Pclose(plist_id);

    /* Open the dataset and get the filespace id */
    dset_id = H5Dopen(file_id, "data", H5P_DEFAULT);
    filespace = H5Dget_space(dset_id);
    dims = DATASIZE;  

    /* Select a hyperslab of the file dataspace */
    counts = localsize;
    offsets = my_id * localsize;
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, &offsets, NULL, &counts,
                        NULL);

    /* Now we can read our local data from the correct position in the
       dataset. Here we use collective read but independent reads are
       also possible. */
    memspace = H5Screate_simple(1, &counts, NULL);
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    status = H5Dread(dset_id, H5T_NATIVE_INT, memspace, filespace,
                      plist_id, localvector);

    /* Close all opened HDF5 handles */
    H5Dclose(dset_id);
    H5Pclose(plist_id);
    H5Sclose(filespace);
    H5Sclose(memspace);
    H5Fclose(file_id);
}
