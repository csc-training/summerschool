#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int arraysize = 100000;
    int msgsize = 100;

    int rank, ntasks;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (ntasks < 2)
    {
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

    // Will hold the number of received elements
    int nrecv = 1;

    // Order of message passing is as follows:
    // 1. rank 0 sends message to rank 1
    // 2. rank 1 receives message from rank 0
    // 3. rank 1 sends message to rank 0
    // 4. rank 0 receives message from rank 1

    // Trying to eg. do both sends before any receives may result in a deadlock
    // depending on how the MPI implementation handles blocking routines.

    if (rank == 0)
    {
        // Send total of 'msgsize' integers to rank 1.
        // Message tag must be valid (>= 0), but we don't use it for anything in this exercise
        int tag = 0;
        MPI_Send(message, msgsize, MPI_INT, 1, tag, MPI_COMM_WORLD);

        MPI_Status status;
        // Receive at most 'msgsize' integers from rank 1 and store info about the received message in 'status'
        MPI_Recv(receiveBuffer, msgsize, MPI_INT, 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        printf("Rank %i received %i elements, first %i\n", rank, msgsize, receiveBuffer[0]);
    }
    else if (rank == 1)
    {
        MPI_Recv(receiveBuffer, msgsize, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(message, msgsize, MPI_INT, 0, 1, MPI_COMM_WORLD);

        printf("Rank %i received %i elements, first %i\n", rank, msgsize, receiveBuffer[0]);
    }
    // If ran with more than 2 processes, the leftover ranks do nothing


    // Free buffers
    free(message);
    free(receiveBuffer);

    MPI_Finalize();
    return 0;
}
