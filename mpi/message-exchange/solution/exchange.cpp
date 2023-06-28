#include <cstdio>
#include <vector>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int myid, ntasks, nrecv;
    constexpr int arraysize = 100000;
    constexpr int msgsize = 100;
    std::vector<int> message(arraysize);
    std::vector<int> receiveBuffer(arraysize);
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // Initialize message and receive buffer
    for (int i = 0; i < arraysize; i++) {
        message[i] = myid;
        receiveBuffer[i] = -1;
    }

    // Send and receive messages as defined in exercise
    if (myid == 0) {
        MPI_Send(message.data(), msgsize, MPI_INT, 1, 1, MPI_COMM_WORLD);
        MPI_Recv(receiveBuffer.data(), arraysize, MPI_INT, 1, 2, MPI_COMM_WORLD,
                 &status);
        MPI_Get_count(&status, MPI_INT, &nrecv);
        printf("Rank %i received %i elements, first %i\n", myid, nrecv, receiveBuffer[0]);
    } else if (myid == 1) {
        MPI_Recv(receiveBuffer.data(), arraysize, MPI_INT, 0, 1, MPI_COMM_WORLD,
                 &status);
        MPI_Send(message.data(), msgsize, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Get_count(&status, MPI_INT, &nrecv);
        printf("Rank %i received %i elements, first %i\n", myid, nrecv, receiveBuffer[0]);
    }

    MPI_Finalize();
    return 0;
}
