#include <cstdio>
#include <vector>
#include <mpi.h>

void print_ordered(double t);

int main(int argc, char *argv[])
{
    int i, rank, ntasks;
    constexpr int size = 10000000;
    std::vector<int> message(size);
    std::vector<int> receiveBuffer(size);
    MPI_Status send_status;
    MPI_Status recv_status;
    MPI_Request send_request;
    MPI_Request recv_request;

    double t0, t1;

    int source, destination, nrecv;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialize buffers
    for (i = 0; i < size; i++) {
        message[i] = rank;
        receiveBuffer[i] = -1;
    }

    // TODO: Set source and destination ranks
    destination = rank + 1;
    source = rank - 1;

    // TODO: Treat boundaries with MPI_PROC_NULL
    if (rank == 0){
        source = MPI_PROC_NULL;
    }
    else if (rank == ntasks - 1){
        destination = MPI_PROC_NULL;
    }

    // Start measuring the time spent in communication
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();

    // TODO: Receive messages
    MPI_Irecv(receiveBuffer.data(), size, MPI_INT, source, rank, MPI_COMM_WORLD, &recv_request);

    // TODO: Send messages
    MPI_Isend(message.data(), size, MPI_INT, destination, rank+1, MPI_COMM_WORLD, &send_request);

    // Wait for MPI_Irecv and MPI_Isend to complete by using requests. Also returns metadata for the communication stored in statuses.
    MPI_Wait(&send_request, &send_status);
    MPI_Wait(&recv_request, &recv_status);

    MPI_Get_count(&recv_status, MPI_INT, &nrecv);

    printf("\nSender: %d. Sent elements: %d. Tag: %d. Receiver: %d\n",
           rank, size, rank + 1, destination);
    printf("Receiver: %d. Received number of elements:%d. First element %d.\n",
           rank, nrecv, receiveBuffer[0]);
    

    // Finalize measuring the time and print it out
    t1 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    fflush(stdout);

    print_ordered(t1 - t0);

    MPI_Finalize();
    return 0;
}

void print_ordered(double t)
{
    int i, rank, ntasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    if (rank == 0) {
        printf("Time elapsed in rank %2d: %6.3f\n", rank, t);
        for (i = 1; i < ntasks; i++) {
            MPI_Recv(&t, 1, MPI_DOUBLE, i, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Time elapsed in rank %2d: %6.3f\n", i, t);
        }
    } else {
        MPI_Send(&t, 1, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD);
    }
}
