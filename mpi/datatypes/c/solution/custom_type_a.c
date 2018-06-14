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

    if (rank == 0) {
        printf("Data in rank 0\n");
        for (i = 0; i < 8; i++) {
            for (j = 0; j < 8; j++) {
                printf("%3d", array[i][j]);
            }
            printf("\n");
        }
    }

    //TODO: Create datatype that describes one column. Use MPI_Type_vector.

    // Create datatype
    MPI_Type_vector(8, 1, 8, MPI_INT, &columntype);
    MPI_Type_commit(&columntype);

    // Send first column of matrix
    if (rank == 0) {
        MPI_Send(&array[0][1], 1, columntype, 1, 1, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Recv(&array[0][1], 1, columntype, 0, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }

    // Print out the result
    if (rank == 1) {
        printf("Received data\n");
        for (i = 0; i < 8; i++) {
            for (j = 0; j < 8; j++) {
                printf("%3d", array[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Type_free(&columntype);
    MPI_Finalize();

    return 0;
}
