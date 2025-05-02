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

        const int sourceRank = 1;

        int messageLength = 1;
        std::vector<int> receiveBuffer(messageLength);

        // Probe for incoming messages from rank 1, without receiving it.
        // MPI_Probe is blocking, causing the calling rank to wait until an incoming message is detected.
        // Info about the message is stored into the MPI_Status variable.
        MPI_Status status;

        MPI_Probe(sourceRank, tag, MPI_COMM_WORLD, &status);

        // Find the number of integers contained in the message that we probed,
        // and store in 'messageLength'
        MPI_Get_count(&status, MPI_INT, &messageLength);

        // Resize the receive buffer so that the full message fits in
        receiveBuffer.resize(messageLength);

        MPI_Recv(receiveBuffer.data(), messageLength, MPI_INT,
            sourceRank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        printf("Rank 0: Received %d integers from rank 1.\n", messageLength);
        // Print the received numbers
        for (int i = 0; i < receiveBuffer.size(); i++ ) {
            printf("receiveBuffer[%d] : %d\n", i, receiveBuffer[i]);
        }

        // For Approach 2, we are told that the maximum 'messageLength' is 10.
        // Therefore the following would work (note the 'count' argument to MPI_Recv):

        /*
        const int maxLength = 10;
        receiveBuffer.resize(maxLength);
        MPI_Status status;
        MPI_Recv(receiveBuffer.data(), maxLength, MPI_INT,
            sourceRank, tag, MPI_COMM_WORLD, &status
        );
        MPI_Get_count(&status, MPI_INT, &messageLength);
        */

        // This is OK for MPI because the receive buffer is always large enough to hold the incoming message.
        // However, this approach potentially wastes memory, and necessitates that we have information about how large the messages can be.
        // Real programs may benefit from a combination of both approaches:
        //  1. Allocate a reusable receive buffer to some initial size
        //  2. Use MPI_Probe to check that the message can fit; if not, reallocate the buffer
        //  3. Receive and wait for the next message
        // This approach can minimize the number of mallocs/frees required for repeated message receiving.
        // Whether it's a good fit for your program depends entirely on the use case.
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
