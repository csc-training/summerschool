#include <cstdio>
#include <vector>
#include <mpi.h>

int main(int argc, char *argv[])
{
    constexpr int arraysize = 100000;
    constexpr int msgsize = 100;
    std::vector<int> message(arraysize);
    std::vector<int> receiveBuffer(arraysize);

    int rank, ntasks;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (ntasks < 2)
    {
        printf("Please run with at least 2 MPI processes\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize message and receive buffer
    for (int i = 0; i < arraysize; i++)
    {
        message[i] = rank;
        receiveBuffer[i] = -1;
    }

    // Order of message passing is as follows:
    // 1. rank 0 sends message to rank 1
    // 2. rank 1 receives message from rank 0
    // 3. rank 1 sends message to rank 0
    // 4. rank 0 receives message from rank 1

    // Trying to eg. do both sends before any receives may result in a deadlock
    // depending on how the MPI implementation handles blocking routines.

    if (rank == 0)
    {
        // Send total of 'msgsize' integers to rank 1.
        // Message tag must be valid (>= 0), but we don't use it for anything in this exercise
        int tag = 0;
        MPI_Send(message.data(), msgsize, MPI_INT, 1, tag, MPI_COMM_WORLD);

        // Receive at most 'msgsize' integers from rank 1
        MPI_Recv(receiveBuffer.data(), msgsize, MPI_INT, 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Rank %i received %i elements, first %i\n", rank, msgsize, receiveBuffer[0]);
    }
    else if (rank == 1)
    {
        MPI_Status status;
        MPI_Recv(receiveBuffer.data(), msgsize, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Send(message.data(), msgsize, MPI_INT, 0, 1, MPI_COMM_WORLD);

        printf("Rank %i received %i elements, first %i\n", rank, msgsize, receiveBuffer[0]);
    }
    // If ran with more than 2 processes, the leftover ranks do nothing


    MPI_Finalize();
    return 0;
}
