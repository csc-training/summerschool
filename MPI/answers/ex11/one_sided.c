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

    /* Create window corresponding to the receive buffer */
    MPI_Win_create(receiveBuffer, sizeof(int) * size, sizeof(int),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &window);

    /* Put messages as defined in exercise */
    MPI_Win_fence(0, window);
    if (myid < ntasks - 1) {
        MPI_Put(message, size, MPI_INT, myid + 1, 0, size, MPI_INT,
                window);
        printf("Origin: %d. Put elements: %d. Target: %d\n", myid, size,
               myid + 1);
    }
    MPI_Win_fence(0, window);

    if (myid > 0) {
        printf("Target: %d. first element %d.\n", myid, receiveBuffer[0]);
    }

    free(message);
    free(receiveBuffer);
    MPI_Win_free(&window);
    MPI_Finalize();
    return 0;
}
