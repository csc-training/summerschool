#include <cstdio>
#include <vector>
#include <mpi.h>

// Solution file to the message-chain exercise. Includes all cases A,B,C


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
    for (int i = 0; i < numElements; i++)
    {
        message[i] = rank;
        receiveBuffer[i] = -1;
    }

    // Set source and destination ranks, handling boundaries with MPI_PROC_NULL
    int source = rank - 1;
    int destination = rank + 1;

    // First rank receives from no one
    if (rank == 0) {
        source = MPI_PROC_NULL;
    }
    // Last rank sends to no one
    if (rank >= ntasks - 1) {
        destination = MPI_PROC_NULL;
    }

    if (rank == 0) {
        printf("\n#### Case A: Chain with MPI_Send and MPI_Recv ####\n");
    }

    // Start measuring the time spent in communication
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Send messages. Note that the send tag must be valid (>= 0) even if our destination is MPI_PROC_NULL
    int sendTag = rank + 1;
    MPI_Send(message.data(), numElements, MPI_INT, destination, sendTag, MPI_COMM_WORLD);

    printf("Sender: %d. Sent elements: %d. Tag: %d. Receiver: %d\n",
           rank, numElements, sendTag, destination);

    // Receive messages
    int recvTag = rank;
    MPI_Recv(receiveBuffer.data(), numElements, MPI_INT, source, recvTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("Receiver: %d. Receive tag: %d. First element: %d.\n",
           rank, recvTag, receiveBuffer[0]);

    // Finalize measuring the time and print it out
    double t1 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    fflush(stdout);

    print_ordered(t1 - t0);

    if (rank == 0) {
        printf("#### End case A ####\n");
    }

    // Repeat the same thing but this time using MPI_Sendrecv

    if (rank == 0) {
        printf("\n#### Case B: Chain with MPI_Sendrecv ####\n");
    }

    t0 = MPI_Wtime();

    sendTag = rank + 1;
    recvTag = rank;
    MPI_Sendrecv(message.data(), numElements, MPI_INT, destination, sendTag,
                 receiveBuffer.data(), numElements, MPI_INT, source, recvTag,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("Sender: %d. Sent elements: %d. Tag: %d. Receiver: %d\n",
           rank, numElements, sendTag, destination);

    printf("Receiver: %d. Receive tag: %d. First element: %d.\n",
        rank, recvTag, receiveBuffer[0]);


    t1 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    fflush(stdout);

    print_ordered(t1 - t0);

    // This MPI_Sendrecv implementation of the message chain is more efficient
    // compared to separately doing MPI_Send and MPI_Recv.
    // This is because the other version does MPI_Send (blocking) first from all ranks
    // except for the last one, so only the last process can proceed to the receive stage.
    // The messages are then received in reverse order, causing the chain to slowly unwind.
    // The first sender (rank 0) has to wait until all other receives have been processed.
    // In contrast, the MPI_Sendrecv version allows rank 0 to receive and continue as soon as
    // rank 1 has finished its send, and so on.

    if (rank == 0) {
        printf("#### End case B ####\n");
    }

    // Repeat again, but this time alternating the sends and receives.
    // Processes with even rank send first, then receive.
    // Processes with odd rank receive first, then send.

    if (rank == 0) {
        printf("\n#### Case C: Chain with alternating send and receive ####\n");
    }

    t0 = MPI_Wtime();

    if (rank % 2 == 0) {
        MPI_Send(message.data(), numElements, MPI_INT, destination, sendTag, MPI_COMM_WORLD);
        MPI_Recv(receiveBuffer.data(), numElements, MPI_INT, source, recvTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(receiveBuffer.data(), numElements, MPI_INT, source, recvTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(message.data(), numElements, MPI_INT, destination, sendTag, MPI_COMM_WORLD);
    }

    printf("Sender: %d. Sent elements: %d. Tag: %d. Receiver: %d\n",
           rank, numElements, sendTag, destination);

    printf("Receiver: %d. Receive tag: %d. First element: %d.\n",
        rank, recvTag, receiveBuffer[0]);

    t1 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    fflush(stdout);

    print_ordered(t1 - t0);

    if (rank == 0) {
        printf("#### End case C ####\n");
    }


    MPI_Finalize();
    return 0;
}

void print_ordered(double t) {
    int rank, ntasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    if (rank == 0) {
        printf("Time elapsed in rank %2d: %6.3f\n", rank, t);
        for (int i = 1; i < ntasks; i++) {
            MPI_Recv(&t, 1, MPI_DOUBLE, i, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Time elapsed in rank %2d: %6.3f\n", i, t);
        }
    } else {
        MPI_Send(&t, 1, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD);
    }
    fflush(stdout);
}
