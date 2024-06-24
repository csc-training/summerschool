#include <stdio.h>
#include <unistd.h>  // for sleep()
#include <mpi.h>

int main(int argc, char *argv[])
{
  int size, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  printf("Hello from rank %d of %d\n", rank, size);

  if (rank == 0) {
    usleep(100);  // "computing"
    double data = 42.0;

    // Send with rank 0
    MPI_Send(&data, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    printf("Rank %d sent %f\n", rank, data);

  } else if (rank == 1) {
    int message_waiting;
    MPI_Status status;
    MPI_Iprobe(0, 0, MPI_COMM_WORLD, &message_waiting, &status);

    while (!message_waiting) {
      printf("Rank %d has no incoming messages. Let's do some computing\n", rank);
      usleep(10);  // "computing"
      MPI_Iprobe(0, 0, MPI_COMM_WORLD, &message_waiting, &status);
    }

    printf("Rank %d has incoming message. Let's receive\n", rank);
    double data;
    MPI_Recv(&data, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    printf("Rank %d received %f\n", rank, data);

  }

  MPI_Finalize();
}
