#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int i, rank, size;
    MPI_Status status;
    int tag=0;
    float data=42.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        for (i=1; i < size; i++) {
            printf("Ping %d..\n", i);
            MPI_Send(&data, 1, MPI_REAL, i, tag, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&data, 1, MPI_REAL, 0, tag, MPI_COMM_WORLD, &status);
        printf("      ..Pong %d\n", rank);
    }

    MPI_Finalize();
    return 0;
}
