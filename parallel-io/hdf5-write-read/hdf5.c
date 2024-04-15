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

    /* Create the dataset */

    /* Select a hyperslab of the file dataspace */

    /* Now we can write our local data to the correct position in the
       dataset. Here we use collective write, but independent writes are
       also possible. */

    /* Close all opened HDF5 handles */
}


void h5_reader(int my_id, int *localvector, int localsize)
{
    herr_t status;
    hid_t plist_id, dset_id, filespace, memspace, file_id;
    hsize_t dims, counts, offsets;

    /* Create the handle for parallel file access property list
       and open a file for reading */

    /* Open the dataset and get the filespace id */

    /* Select a hyperslab of the file dataspace */

    /* Now we can read our local data from the correct position in the
       dataset. Here we use collective read but independent reads are
       also possible. */

    /* Close all opened HDF5 handles */
}
