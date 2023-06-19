#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank, ntasks;
    int array[8][6];
    int recvarray[8][6];
    MPI_Datatype vector;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    // Initialize arrays
    if (rank == 0) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 6; j++) {
                array[i][j] = (i + 1) * 10 + j + 1;
            }
        }
    } else {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 6; j++) {
                array[i][j] = 0;
            }
        }
    }
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 6; j++) {
            recvarray[i][j] = 0;
        }
    }

    // Print data on rank 0
    if (rank == 0) {
        printf("Data on rank %d\n", rank);
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 6; j++) {
                printf("%3d", array[i][j]);
            }
            printf("\n");
        }
    }

    // Create datatype
    MPI_Type_vector(8, 1, 6, MPI_INT, &vector);
    MPI_Type_create_resized(vector, 0, sizeof(int), &vector);
    MPI_Type_commit(&vector);

    // Scatter columns
    MPI_Scatter(array, 1, vector,
                recvarray, 1, vector, 0, MPI_COMM_WORLD);

    // Free datatype
    MPI_Type_free(&vector);

    // Print received data
    for (int irank = 0; irank < ntasks; irank++) {
        if (irank == rank) {
            printf("Received data on rank %d\n", rank);
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 6; j++) {
                    printf("%3d", recvarray[i][j]);
                }
                printf("\n");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}
