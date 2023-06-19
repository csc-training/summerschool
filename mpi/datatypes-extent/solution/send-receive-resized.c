#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank;
    int array[8][6];
    MPI_Datatype vector;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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
    MPI_Aint lb, extent;
    MPI_Type_get_extent(vector, &lb, &extent);
    if (rank == 0) {
        printf("Extent before resize is %ld elements\n", extent / sizeof(int));
    }

    // Resize datatype
    MPI_Datatype tmp = vector;
    MPI_Type_create_resized(tmp, 0, sizeof(int), &vector);
    MPI_Type_get_extent(vector, &lb, &extent);
    if (rank == 0) {
        printf("Extent after  resize is %ld elements\n", extent / sizeof(int));
    }
    MPI_Type_commit(&vector);

    // Send data from rank 0 to rank 1
    if (rank == 0) {
        MPI_Send(array, 2, vector, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Recv(array, 2, vector, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Free datatype
    MPI_Type_free(&vector);

    // Print received data
    if (rank == 1) {
        printf("Received data on rank %d\n", rank);
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 6; j++) {
                printf("%3d", array[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();

    return 0;
}
