#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int i, myid, ntasks;
    int msgsize = 10000000;
    int *message;
    int *receiveBuffer;
    MPI_Status status;

    double t0, t1;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    /* Allocate message buffers */
    message = (int *)malloc(sizeof(int) * msgsize);
    receiveBuffer = (int *)malloc(sizeof(int) * msgsize);
    /* Initialize message */
    for (i = 0; i < msgsize; i++) {
        message[i] = myid;
    }

    /* Start measuring the time spent in communication */
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();

    /* Send and receive messages */
    if ((myid > 0) && (myid < ntasks - 1)) {
        MPI_Sendrecv(message, msgsize, MPI_INT, myid + 1, myid + 1,
                     receiveBuffer, msgsize, MPI_INT, myid - 1, MPI_ANY_TAG,
                     MPI_COMM_WORLD, &status);
        printf("Sender: %d. Sent elements: %d. Tag: %d. Receiver: %d\n",
               myid, msgsize, myid + 1, myid + 1);
    }
    /* Only send a message */
    else if (myid < ntasks - 1) {
        MPI_Send(message, msgsize, MPI_INT, myid + 1, myid + 1,
                 MPI_COMM_WORLD);
        printf("Sender: %d. Sent elements: %d. Tag: %d. Receiver: %d\n",
               myid, msgsize, myid + 1, myid + 1);
    }
    /* Only receive a message */
    else if (myid > 0) {
        MPI_Recv(receiveBuffer, msgsize, MPI_INT, myid - 1, myid,
                 MPI_COMM_WORLD, &status);
        printf("Receiver: %d. first element %d.\n",
               myid, receiveBuffer[0]);
    }

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
