#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank;
    int sendarray[8][6];
    int recvarray[8][6];
    MPI_Datatype vector, vector2;


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialize arrays
    if (rank == 0) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 6; j++) {
                sendarray[i][j] = (i + 1) * 10 + j + 1;
            }
        }
    }

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 6; j++) {
            recvarray[i][j] = 0;
        }
    }

    if (rank == 0) {
        printf("Data in rank 0\n");
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 6; j++) {
                printf("%3d", sendarray[i][j]);
            }
            printf("\n");
        }
    }

    // TODO create datatype

    // Communicate with the datatype
    if (rank == 0)

    else if (rank == 1)


    // free datatype

    // TODO end

    if (rank == 1) {
        printf("Received data\n");
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 6; j++) {
                printf("%3d", recvarray[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();

    return 0;
}
