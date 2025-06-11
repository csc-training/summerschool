#include <cstdio>
#include <vector>
#include <mpi.h>

void init_buffer(std::vector<int> &buffer);
void print_buffer(std::vector<int> &buffer);


int main(int argc, char *argv[])
{
    int ntasks, rank, size=12;
    std::vector<int> buf(size);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Initialize message buffer */
    init_buffer(buf);

    /* Print data that will be sent */
    print_buffer(buf);

    /* Start timing */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* Send everywhere */
    // TODO: Implement the broadcast of the array buf

    /* End timing */
    double t1 = MPI_Wtime();

    /* Print data that was received */
    print_buffer(buf);
    if (rank == 0) {
        printf("Time elapsed: %6.8f s\n", t1 - t0);
    }

    MPI_Finalize();
    return 0;
}


void init_buffer(std::vector<int> &buffer)
{
    int rank;
    int buffersize = buffer.size();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        for (int i = 0; i < buffersize; i++) {
            buffer[i] = i;
        }
    } else {
        for (int i = 0; i < buffersize; i++) {
            buffer[i] = -1;
        }
    }
}


void print_buffer(std::vector<int> &buffer)
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
