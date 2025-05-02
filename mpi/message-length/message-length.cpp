#include <cstdio>
#include <vector>
#include <random>
#include <cassert>

#include <mpi.h>

int randomMessageLength();

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, ntasks;
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (ntasks < 2) {
        printf("Please run with at least 2 MPI processes\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // just some tag for the message
    const int tag = 123;

    if (rank == 1) {

        // Generate random message size in rank 1 only. Other ranks do not know the size
        const int messageLength = randomMessageLength();
        std::vector<int> message(messageLength);

        // fill in a test message: element i has value i
        for (int i = 0; i < message.size(); i++) {
            message[i] = i;
        }

        // Send the test message to rank 0
        printf("Rank 1: Sending %d integers to rank 0\n", messageLength);
        fflush(stdout);
        MPI_Send(message.data(), message.size(), MPI_INT, 0, tag, MPI_COMM_WORLD);
    }

    if (rank == 0) {

        int messageLength = 1;
        std::vector<int> receiveBuffer(messageLength);

        // TODO: receive the full message sent from rank 1.
        // Use MPI_Probe and MPI_Get_count to figure out the number of integers in the message.
        // Store this count in 'messageLength', then reallocate 'receiveBuffer'
        // to correct size, and finally receive the message.

        const int sourceRank = 1;

        // ... your code here ...

        // Receive the message. Will error with MPI_ERR_TRUNCATE if the buffer is too small for the incoming message
        MPI_Recv(receiveBuffer.data(), receiveBuffer.size(), MPI_INT,
            sourceRank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        printf("Rank 0: Received %d integers from rank 1.\n", messageLength);
        // Print the received numbers
        for (int i = 0; i < receiveBuffer.size(); i++ ) {
            printf("receiveBuffer[%d] : %d\n", i, receiveBuffer[i]);
        }

    }

    MPI_Finalize();
    return 0;
}


// Helper functions below, no need to modify these

// Generate random int in specified range (inclusive), using Mersenne Twister
int randomInt(int min, int max) {

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution(min, max);
    return distribution(gen);
}

// Generates a random message length for the test message
int randomMessageLength() {

    int res = randomInt(2, 10);
    assert(res > 0 && "Can't happen: generated nonsensical message length...");

    return res;
}
