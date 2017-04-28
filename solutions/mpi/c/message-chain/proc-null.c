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
    printf("Sender: %d. Sent elements: %d. Tag: %d. Receiver: %d\n",
            myid, size, myid + 1, destination);

    free(message);
    free(receiveBuffer);
    MPI_Finalize();
    return 0;
}
