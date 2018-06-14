#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int i, myid, ntasks;
    int size = 10000000;
    int *message;
    int *receiveBuffer;
    MPI_Status statuses[2];
    MPI_Request requests[2];

    double t0, t1;

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

    /* Start measuring the time spend in communication */
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();

    /* Receive messages in the back ground */
    MPI_Irecv(receiveBuffer, size, MPI_INT, source, MPI_ANY_TAG,
              MPI_COMM_WORLD, &requests[0]);
    /* Send messages in the back ground */
    MPI_Isend(message, size, MPI_INT, destination, myid + 1,
              MPI_COMM_WORLD, &requests[1]);

    /* Blocking wait for messages */
    MPI_Waitall(2, requests, statuses);

    /* Use status parameter to find out the no. of elements received */
    MPI_Get_count(&statuses[0], MPI_INT, &count);
    printf("Sender: %d. Sent elements: %d. Tag: %d. Receiver: %d\n",
           myid, size, myid + 1, destination);
    printf("Receiver: %d. Received elements: %d. Tag %d. Sender: %d.\n",
           myid, count, statuses[0].MPI_TAG, statuses[0].MPI_SOURCE);

    /* Finalize measuring the time and print it out */
    t1 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    fflush(stdout);

    printf("Time elapsed in rank %2d: %6.3f\n", myid, t1 - t0);

    free(message);
    free(receiveBuffer);
    MPI_Finalize();
    return 0;
}
