#include <cstdio>
#include <vector>
#include <mpi.h>

#define MAX_PRINT_SIZE 12

void init_buffer(std::vector<int> &buffer);
void print_buffer(std::vector<int> &buffer);


int main(int argc, char *argv[])
{
    int size, rank, buf_size=12;
    std::vector<int> buf(buf_size);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Initialize message buffer */
    init_buffer(buf);

    /* Print data that will be sent */
    print_buffer(buf);

    /* Start timing */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* Send everywhere */
    // Distributing data in a tree
    if (rank % 2 == 0) {
        int block_size; // size of this rank's block where to distribute data
        if (rank == 0) {
            // Rank 0 only sends
            block_size = 1;
            while (block_size < size) {
                block_size <<= 1;
            }
        } else {
            // Other even ranks first receive and then send to their block
            int source = 0;

            while (true) {
                block_size = 2;
                int block_rank = rank - source;
                while (block_rank > block_size) {
                    block_size <<= 1;
                }
                if (block_rank == block_size) {
                    break;
                } else {
                    source += block_size >> 1;
                }
            }

            // printf("Rank %02d recv from %02d\n", rank, source); fflush(stdout);
            MPI_Recv(buf.data(), buf_size, MPI_INT, source, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Send to the power of 2 destinations in the block
        // For example, rank 0 osends to ranks 2, 4, 8, 16, 32, ... (in reverse order)
        for (int block_dest = block_size >> 1; block_dest > 0; block_dest >>= 1) {
            int dest = rank + block_dest;
            if (dest < size) { // guard for sizes not power of 2
                // printf("Rank %02d send to   %02d\n", rank, dest); fflush(stdout);
                MPI_Send(buf.data(), buf_size, MPI_INT, dest, 123, MPI_COMM_WORLD);
            }
        }
    } else {
        // Odd ranks receive from previous rank (which is even)
        int source = rank - 1;
        // printf("Rank %02d recv from %02d\n", rank, source); fflush(stdout);
        MPI_Recv(buf.data(), buf_size, MPI_INT, source, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

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
    int rank, size;
    int buffersize = buffer.size();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> printbuffer(buffersize * size);

    MPI_Gather(buffer.data(), buffersize, MPI_INT,
               printbuffer.data(), buffersize, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int j = 0; j < size; j++) {
            printf("Task %2i:", j);
            for (int i = 0; i < MAX_PRINT_SIZE; i++) {
                printf(" %2i", printbuffer[i + buffersize * j]);
            }
            if (MAX_PRINT_SIZE < buffersize) {
                printf(" ...");
            }
            printf("\n");
        }
        printf("\n");
    }
}
