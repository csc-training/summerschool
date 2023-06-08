#include <cstdio>
#include <vector>
#include <mpi.h>

void print_buffers(int *printbuffer, int *sendbuffer, int buffersize);
void init_buffers(int *sendbuffer, int *recvbuffer, int buffersize);


int main(int argc, char *argv[])
{
    int ntasks, myid, size=12;
    std::vector<int> sendbuf(size);
    std::vector<int> recvbuf(size);
    std::vector<int> printbuf(size*size);
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    /* Initialize message buffers */
    init_buffers(sendbuf.data(), recvbuf.data(), size);

    /* Print data that will be sent */
    print_buffers(printbuf.data(), sendbuf.data(), size);

    /* Send everywhere */
    if (size % ntasks != 0) {
        if (myid == 0) {
            fprintf(stderr, "Size not divisible by the number of tasks. This program will fail.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    int block_size = size/ntasks;
    if (myid == 0) {
        for (int i = 1; i < ntasks; i++) {
            MPI_Send(&sendbuf[i*block_size], block_size, MPI_INT, i, i, MPI_COMM_WORLD);
        }

        // Scatter also the local part
        for (int i = 0; i < block_size; i++) {
            recvbuf[i] = sendbuf[i];
        }
    } else {
        MPI_Recv(recvbuf.data(), block_size, MPI_INT, 0, myid, MPI_COMM_WORLD, &status);
    }

    /* Print data that was received */
    print_buffers(printbuf.data(), recvbuf.data(), size);

    MPI_Finalize();
    return 0;
}


void init_buffers(int *sendbuffer, int *recvbuffer, int buffersize)
{
    int rank, i;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        for (i = 0; i < buffersize; i++) {
            recvbuffer[i] = -1;
            sendbuffer[i] = i;
        }
    } else {
        for (i = 0; i < buffersize; i++) {
            recvbuffer[i] = -1;
            sendbuffer[i] = -1;
        }
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
            printf("Task %2i:", j);
            for (i = 0; i < buffersize; i++) {
                printf(" %2i", printbuffer[i + buffersize * j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
