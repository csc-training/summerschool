#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define NTASKS 4

void print_buffers(int *printbuffer, int *sendbuffer, int buffersize);
void init_buffers(int *sendbuffer, int *recvbuffer, int buffersize);


int main(int argc, char *argv[])
{
    int ntasks, rank, color;
    int sendbuf[2 * NTASKS], recvbuf[2 * NTASKS];
    int printbuf[2 * NTASKS * NTASKS];

    int offsets[NTASKS] = { 0, 1, 2, 4 };
    int counts[NTASKS] = { 1, 1, 2, 4 };
    MPI_Request request;

    MPI_Comm sub_comm;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (ntasks != NTASKS) {
        if (rank == 0) {
            fprintf(stderr, "Run this program with %i tasks.\n", NTASKS);
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    /* First collective operation to send (0,1,2,...,7) everywhere */
    /* Initialize sendbuf and broadcast */
    init_buffers(sendbuf, recvbuf, 2 * NTASKS);
    MPI_Ibcast(sendbuf, 2 * NTASKS, MPI_INT, 0, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    print_buffers(printbuf, sendbuf, 2 * NTASKS);

    /* Initialize buffers for a) */
    init_buffers(sendbuf, recvbuf, 2 * NTASKS);
    print_buffers(printbuf, sendbuf, 2 * NTASKS);

    /* Scatter the elements from task 0 */
    MPI_Iscatter(sendbuf, 2, MPI_INT, recvbuf, 2, MPI_INT, 0,
                 MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    print_buffers(printbuf, recvbuf, 2 * NTASKS);

    /* Gather varying size data to task 1 */
    init_buffers(sendbuf, recvbuf, 2 * NTASKS);
    MPI_Igatherv(sendbuf, counts[rank], MPI_INT, recvbuf, counts,
                 offsets, MPI_INT, 1, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    print_buffers(printbuf, recvbuf, 2 * NTASKS);

    /* Create new communicator and reduce the data */
    init_buffers(sendbuf, recvbuf, 2 * NTASKS);
    if (rank / 2 == 0) {
        color = 1;
    } else {
        color = 2;
    }
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &sub_comm);
    MPI_Ireduce(sendbuf, recvbuf, 2 * NTASKS, MPI_INT, MPI_SUM, 0,
                sub_comm, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    print_buffers(printbuf, recvbuf, 2 * NTASKS);

    MPI_Finalize();
    return 0;
}


void init_buffers(int *sendbuffer, int *recvbuffer, int buffersize)
{
    int rank, i;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    for (i = 0; i < buffersize; i++) {
        recvbuffer[i] = -1;
        sendbuffer[i] = i + buffersize * rank;
    }
}


void print_buffers(int *printbuffer, int *sendbuffer, int buffersize)
{
    int i, j, rank, ntasks;

    MPI_Gather(sendbuffer, buffersize, MPI_INT,
               printbuffer, buffersize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    if (rank == 0) {
        for (j = 0; j < ntasks; j++) {
            printf("Task %i:", j);
            for (i = 0; i < buffersize; i++) {
                printf(" %2i", printbuffer[i + buffersize * j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
