#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int i, myid, ntasks;
    int size = 100;
    int *message;
    int *receiveBuffer;
    MPI_Status status;

    int source, destination;
    int count;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    /* Allocate message buffers */
    message = malloc(sizeof(int) * size);
    receiveBuffer = malloc(sizeof(int) * size);
    /* Initialize message */
    for (i = 0; i < size; i++) {
        message[i] = myid;
    }

    /* Set source and destination ranks */
    if (myid < ntasks - 1) {
        destination = myid + 1;
    } else {
        destination = MPI_PROC_NULL;
    }
    if (myid > 0) {
        source = myid - 1;
    } else {
        source = MPI_PROC_NULL;
    }

    /* Send and receive messages */
    MPI_Sendrecv(message, size, MPI_INT, destination, myid + 1,
                 receiveBuffer, size, MPI_INT, source, MPI_ANY_TAG,
                 MPI_COMM_WORLD, &status);
    /* Use status parameter to find out the no. of elements received */
    MPI_Get_count(&status, MPI_INT, &count);
    printf("Sender: %d. Sent elements: %d. Tag: %d. Receiver: %d\n",
            myid, size, myid + 1, destination);
    printf("Receiver: %d. Received elements: %d. Tag %d. Sender: %d.\n",
            myid, count, status.MPI_TAG, status.MPI_SOURCE);

    free(message);
    free(receiveBuffer);
    MPI_Finalize();
    return 0;
}
