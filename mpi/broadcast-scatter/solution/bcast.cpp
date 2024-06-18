#include <cstdio>
#include <vector>
#include <mpi.h>

void init_buffers(std::vector<int> &sendbuffer, std::vector<int> &recvbuffer);
void print_buffers(std::vector<int> &buffer);


int main(int argc, char *argv[])
{
    int ntasks, rank, size=12;
    std::vector<int> sendbuf(size);
    std::vector<int> recvbuf(size);
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Initialize message buffers */
    init_buffers(sendbuf, recvbuf);

    /* Print data that will be sent */
    print_buffers(sendbuf);

    /* Send everywhere */
    if (rank == 0) {
        for (int i = 1; i < ntasks; i++) {
            MPI_Send(sendbuf.data(), size, MPI_INT, i, i, MPI_COMM_WORLD);
        }

        // Broadcast also the local part
        // Note: The real MPI_Bcast() function uses the same buffer for send and recv!
        for (int i = 0; i < size; i++) {
            recvbuf[i] = sendbuf[i];
        }
    } else {
        MPI_Recv(recvbuf.data(), size, MPI_INT, 0, rank, MPI_COMM_WORLD, &status);
    }

    /* Print data that was received */
    print_buffers(recvbuf);

    MPI_Finalize();
    return 0;
}


void init_buffers(std::vector<int> &sendbuffer, std::vector<int> &recvbuffer)
{
    int rank;
    int buffersize = sendbuffer.size();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        for (int i = 0; i < buffersize; i++) {
            recvbuffer[i] = -1;
            sendbuffer[i] = i;
        }
    } else {
        for (int i = 0; i < buffersize; i++) {
            recvbuffer[i] = -1;
            sendbuffer[i] = -1;
        }
    }
}


void print_buffers(std::vector<int> &buffer)
{
    int rank, ntasks;
    int buffersize = buffer.size();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    std::vector<int> printbuffer(buffersize * ntasks);

    MPI_Gather(buffer.data(), buffersize, MPI_INT,
               printbuffer.data(), buffersize, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int j = 0; j < ntasks; j++) {
            printf("Task %2i:", j);
            for (int i = 0; i < buffersize; i++) {
                printf(" %2i", printbuffer[i + buffersize * j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
