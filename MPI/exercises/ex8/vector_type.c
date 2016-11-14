#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank;
    int array[8][8];
    MPI_Datatype columntype;

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
    // Create datatype

    // Send first column of matrix
    if (rank == 0) {
        MPI_Send( , , , 1, 1, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Recv( , , , 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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

    /* TODO: Free datatype*/
    MPI_Finalize();

    return 0;
}
