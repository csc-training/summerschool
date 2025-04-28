#include <cstdio>
#include <vector>
#include <mpi.h>

void print_ordered(double t);

int main(int argc, char *argv[])
{
    constexpr int numElements = 10000000;
    std::vector<int> message(numElements);
    std::vector<int> receiveBuffer(numElements);

    MPI_Init(&argc, &argv);

    int rank, ntasks;
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialize buffers
    for (int i = 0; i < numElements; i++) {
        message[i] = rank;
        receiveBuffer[i] = -1;
    }

    // TODO: Set source and destination ranks to form a message chain:
    //      rank1 -> rank2 -> rank3 -> ...
    // Treat boundaries with MPI_PROC_NULL.
    int source = -1;
    int destination = -1;

    // Start measuring the time spent in communication
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // TODO: Send messages


    printf("Sender: %d. Sent elements: %d. Tag: %d. Receiver: %d\n",
           rank, numElements, rank + 1, destination
    );

    // TODO: Receive messages


    printf("Receiver: %d. first element %d\n", rank, receiveBuffer[0]);


    // Finalize measuring the time and print it out
    double t1 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    fflush(stdout);

    print_ordered(t1 - t0);

    MPI_Finalize();
    return 0;
}

// Function that prints out timings. No need to modify this
void print_ordered(double t)
{
    int rank, ntasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    if (rank == 0)
    {
        printf("Time elapsed in rank %2d: %6.3f\n", rank, t);
        for (int i = 1; i < ntasks; i++) {
            MPI_Recv(&t, 1, MPI_DOUBLE, i, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Time elapsed in rank %2d: %6.3f\n", i, t);
        }
    }
    else
    {
        MPI_Send(&t, 1, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD);
    }
}
