#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>


int main(int argc, char* argv[]) {
    int ntasks, my_id, irank;
    int dims[2];      /* Dimensions of the grid */
    int coords[2];    /* Coordinates in the grid */
    int neighbors[4]; /* Neighbors in 2D grid */
    int period[2] = {1, 1};
    MPI_Comm comm2d;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

    /* Determine the process grid (dims[0] x dims[1] = ntasks) */
    if (ntasks < 16) {
        dims[0] = 2;
    } else if (ntasks >= 16 && ntasks < 64) {
        dims[0] = 4;
    } else if (ntasks >= 64 && ntasks < 256) {
        dims[0] = 8;
    } else {
        dims[0] = 16;
    }
    dims[1] = ntasks / dims[0];

    if (dims[0] * dims[1] != ntasks) {
        fprintf(stderr, "Incompatible dimensions: %i x %i != %i\n",
                dims[0], dims[1], ntasks);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    /* Create the 2D Cartesian communicator */
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, 1, &comm2d);
    /* Find out and store the ranks with which to perform halo exchange */
    MPI_Cart_shift(comm2d, 0, 1, &neighbors[0], &neighbors[1]);
    MPI_Cart_shift(comm2d, 1, 1, &neighbors[2], &neighbors[3]);
    /* Find out and store also the Cartesian coordinates of a rank */
    MPI_Cart_coords(comm2d, my_id, 2, coords);

    for (irank = 0; irank < ntasks; irank++) {
        if (my_id == irank) {
            printf("%3i = %2i %2i neighbors=%3i %3i %3i %3i\n",
                   my_id, coords[0], coords[1], neighbors[0], neighbors[1],
                   neighbors[2], neighbors[3]);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
