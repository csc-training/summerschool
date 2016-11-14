#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>


int main(int argc, char *argv[])
{
    int i, myid, ntasks;
    int size = 100;
    int *message;
    int *receiveBuffer;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    /* Allocate message */
    message = malloc(sizeof(int) * size);
    receiveBuffer = malloc(sizeof(int) * size);
    /* Initialize message */
    for (i = 0; i < size; i++)
        message[i] = myid;

    /* TODO start */
    /* Send and receive messages as defined in exercise */
    if (myid < ntasks - 1) {

        printf("Sender: %d. Sent elements: %d. Tag: %d. Receiver: %d\n",
               myid, size, myid + 1, myid + 1);
    }

    if (myid > 0) {

        printf("Receiver: %d. first element %d.\n",
               myid, receiveBuffer[0]);
    }

    /* TODO end */

    free(message);
    free(receiveBuffer);
    MPI_Finalize();
    return 0;
}
