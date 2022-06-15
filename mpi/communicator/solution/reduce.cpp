#include <cstdio>
#include <array>
#include <mpi.h>

#define NTASKS 4

template<size_t N, size_t N2>
void print_buffers(std::array<int, N2> &printbuffer, std::array<int, N> &sendbuffer);
template<size_t N>
void init_buffers(std::array<int, N> &sendbuffer, std::array<int, N> &recvbuffer);


int main(int argc, char *argv[])
{
    int ntasks, rank, color;
    std::array<int, 2 * NTASKS> sendbuf, recvbuf;
    std::array<int, 2 * NTASKS * NTASKS> printbuf;

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

    // Initialize message buffers 
    init_buffers(sendbuf, recvbuf);

    // Print data that will be sent */
    print_buffers(printbuf, sendbuf);

    /* TODO: use a single collective communication call (and maybe prepare
     *       some parameters for the call) */
    /* Create new communicator and reduce the data */
    if (rank / 2 == 0) {
        color = 1;
    } else {
        color = 2;
    }
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &sub_comm);
    MPI_Reduce(sendbuf.data(), recvbuf.data(), sendbuf.size(), MPI_INT, MPI_SUM, 0,
               sub_comm);


    /* Print data that was received */
    /* TODO: add correct buffer */
    print_buffers(printbuf, recvbuf);

    MPI_Finalize();
    return 0;
}


template<size_t N>
void init_buffers(std::array<int, N> &sendbuffer, std::array<int, N> &recvbuffer)
{
    int rank;

    const int buffersize = sendbuffer.size();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    for (int i = 0; i < buffersize; i++) {
        recvbuffer[i] = -1;
        sendbuffer[i] = i + buffersize * rank;
    }
}


template<size_t N, size_t N2>
void print_buffers(std::array<int, N2> &printbuffer, std::array<int, N> &sendbuffer)
{
    int rank, ntasks;

    const int buffersize = sendbuffer.size();

    MPI_Gather(sendbuffer.data(), buffersize, MPI_INT,
               printbuffer.data(), buffersize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    if (rank == 0) {
        for (int j = 0; j < ntasks; j++) {
            printf("Task %i:", j);
            for (int i = 0; i < buffersize; i++) {
                printf(" %2i", printbuffer[i + buffersize * j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
