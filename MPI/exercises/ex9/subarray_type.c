#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank;
    int array[8][8];
    MPI_Datatype blocktype;
    int sizes[2], subsizes[2], offsets[2];

    int i, j;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialize arrays
    if (rank == 0) {
        for (i = 0; i < 8; i++) {
            for (j = 0; j < 8; j++) {
                array[i][j] = (i + 1) * 10 + j + 1;
            }
        }
    } else {
        for (i = 0; i < 8; i++) {
            for (j = 0; j < 8; j++) {
                array[i][j] = 0;
            }
        }
    }

    /* TODO start */
    // Create datatype for a subblock [2:5][3:5] of the 8x8 matrix
    sizes[0] = sizes[1] = ;
    subsizes[0] = ;
    subsizes[1] = ;
    offsets[0] = ;
    offsets[1] = ;

    // Send a block of a matrix using the user-defined datatype
    if (rank == 0) {
        MPI_Send( ,  ,  , 1, 1, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Recv( ,  ,  , 0, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }

    /* TODO end */

    // Print out the result
    if (rank == 1) {
        for (i = 0; i < 8; i++) {
            for (j = 0; j < 8; j++) {
                printf("%3d", array[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Type_free(&blocktype);
    MPI_Finalize();

    return 0;
}
