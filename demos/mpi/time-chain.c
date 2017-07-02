#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int i, myid, ntasks;
    int size = 10000000;
    int *message;
    int *receiveBuffer;
    MPI_Status status, statuses[2];
    MPI_Request requests[2];

    double t0, t1, tmax=0, time=0;

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

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();

    /* Send+Recv */
    MPI_Recv(receiveBuffer, size, MPI_INT, source, MPI_ANY_TAG,
              MPI_COMM_WORLD, &status);
    MPI_Send(message, size, MPI_INT, destination, myid + 1,
              MPI_COMM_WORLD);

    t1 = MPI_Wtime() - t0;
    MPI_Reduce(&t1, &time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t1, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (myid == 0) {
        time = time / ntasks;
        printf("  Send+Recv:  avg %6.3f s / max %6.3f s\n", time, tmax);
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();

    /* Sendrecv */
    MPI_Sendrecv(message, size, MPI_INT, destination, myid + 1,
                 receiveBuffer, size, MPI_INT, source, MPI_ANY_TAG,
                 MPI_COMM_WORLD, &status);

    t1 = MPI_Wtime() - t0;
    MPI_Reduce(&t1, &time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t1, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (myid == 0) {
        time = time / ntasks;
        printf("   Sendrecv:  avg %6.3f s / max %6.3f s\n", time, tmax);
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();

    /* Isend+Irecv */
    MPI_Irecv(receiveBuffer, size, MPI_INT, source, MPI_ANY_TAG,
              MPI_COMM_WORLD, &requests[0]);
    MPI_Isend(message, size, MPI_INT, destination, myid + 1,
              MPI_COMM_WORLD, &requests[1]);
    MPI_Waitall(2, requests, statuses);

    t1 = MPI_Wtime() - t0;
    MPI_Reduce(&t1, &time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t1, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (myid == 0) {
        time = time / ntasks;
        printf("Isend+Irecv:  avg %6.3f s / max %6.3f s\n\n", time, tmax);
        fflush(stdout);
    }

    free(message);
    free(receiveBuffer);
    MPI_Finalize();
    return 0;
}
