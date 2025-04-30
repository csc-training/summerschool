#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {

    int arraysize = 100000;
    int msgsize = 100;

    int rank, ntasks;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (ntasks < 2) {
        printf("Please run with at least 2 MPI processes\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Allocate buffers
    int* message = (int *)malloc(sizeof(int) * arraysize);
    int* receiveBuffer = (int *)malloc(sizeof(int) * arraysize);

    // Initialize message and receive buffer
    for (int i = 0; i < arraysize; i++) {
        message[i] = rank;
        receiveBuffer[i] = -1;
    }

    // TODO: Implement sending and receiving as defined in the assignment,
    // using MPI_Send and MPI_Recv functions.
    // Send 'msgsize' integers from the array "message",
    // and receive the same number of integers into "receiveBuffer".
    // You may hardcode the message passing to happen between ranks 0 and 1.

    if (rank == 0) {

        // ... your code here ...

        printf("Rank %i received %i elements, first %i\n", rank, msgsize, receiveBuffer[0]);
    }
    else if (rank == 1) {

        // .. your code here ...

        printf("Rank %i received %i elements, first %i\n", rank, msgsize, receiveBuffer[0]);
    }


    // Free buffers
    free(message);
    free(receiveBuffer);

    MPI_Finalize();
    return 0;
}
