#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int myid, ntasks, nrecv;
    int arraysize = 100000;
    int msgsize = 100;
    int *message;
    int *receiveBuffer;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // Allocate buffers
    message = (int *)malloc(sizeof(int) * arraysize);
    receiveBuffer = (int *)malloc(sizeof(int) * arraysize);

    // Initialize message and receive buffer
    for (int i = 0; i < arraysize; i++) {
        message[i] = myid;
        receiveBuffer[i] = -1;
    }

    // Send and receive messages as defined in exercise
    if (myid == 0) {
        MPI_Send(message, msgsize, MPI_INT, 1, 1, MPI_COMM_WORLD);
        MPI_Recv(receiveBuffer, msgsize, MPI_INT, 1, 2, MPI_COMM_WORLD,
                 &status);
        MPI_Get_count(&status, MPI_INT, &nrecv);
        printf("Rank %i received %i elements, first %i\n", myid, nrecv, receiveBuffer[0]);
    } else if (myid == 1) {
        MPI_Recv(receiveBuffer, msgsize, MPI_INT, 0, 1, MPI_COMM_WORLD,
                 &status);
        MPI_Send(message, msgsize, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Get_count(&status, MPI_INT, &nrecv);
        printf("Rank %i received %i elements, first %i\n", myid, nrecv, receiveBuffer[0]);
    }

    // Free buffers
    free(message);
    free(receiveBuffer);

    MPI_Finalize();
    return 0;
}
