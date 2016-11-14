#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>


int main(int argc, char *argv[])
{
    int i, myid, ntasks;
    int size = 100;
    int *message;
    int *receiveBuffer;
    MPI_Win window;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    /* Allocate message */
    message = malloc(sizeof(int) * size);
    receiveBuffer = malloc(sizeof(int) * size);
    /* Initialize message */
    for (i = 0; i < size; i++) {
        message[i] = myid;
    }

    /* TODO: Create a MPI window for the receive buffer and then
             use one-sided communication routines to put correct
             values to target process buffers as defined in the
             message chain exercise */

    /* Create window corresponding to the receive buffer */

    /* Put messages as defined in exercise */
    /* TODO: Add synchronization! */

    if (myid < ntasks - 1) {

        /* TODO: communicate */
        printf("Origin: %d. Put elements: %d. Target: %d\n", myid, size,
               myid + 1);
    }

    if (myid > 0) {
        printf("Target: %d. first element %d.\n", myid, receiveBuffer[0]);
    }

    free(message);
    free(receiveBuffer);
    MPI_Win_free(&window);
    MPI_Finalize();
    return 0;
}
