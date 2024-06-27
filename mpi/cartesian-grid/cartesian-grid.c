#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>


int main(int argc, char* argv[]) {
    int ntasks, rank, irank;
    int dims[2] = {0};      /* Dimensions of the grid */
    int coords[2] = {0};    /* Coordinates in the grid */
    int neighbors[4] = {0}; /* Neighbors in 2D grid */
    int period[2] = {1, 1};
    int num_dims = 2;
    int reorder = 0;
    MPI_Comm comm2d;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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
    /* TODO */
    MPI_Cart_create(MPI_COMM_WORLD, num_dims, dims, period, reorder, &comm2d);

    /* Find out and store the neighboring ranks */
    /* TODO */
    MPI_Cart_shift(comm2d, 0, 1, &neighbors[0], &neighbors[1]);  // up and down
    MPI_Cart_shift(comm2d, 1, 1, &neighbors[2], &neighbors[3]);  // left and right

    //MPI_Cart_shift(MPI_Comm comm, int direction, int disp,
    //int *rank_source, int *rank_dest)

    /* Find out and store also the Cartesian coordinates of a rank */
    /* TODO */
    MPI_Cart_coords(comm2d, rank, 2, coords);

    for (irank = 0; irank < ntasks; irank++) {
        if (rank == irank) {
            printf("%3i = %2i %2i neighbors= up: %3i down: %3i left: %3i right: %3i\n",
                   rank, coords[0], coords[1], neighbors[0], neighbors[1],
                   neighbors[2], neighbors[3]);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
