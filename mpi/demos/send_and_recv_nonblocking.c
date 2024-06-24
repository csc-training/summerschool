#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
  int size, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  printf("Hello from rank %d of %d\n", rank, size);

  if (rank == 0) {
    // Send with rank 0
    double data = 42.0;
    MPI_Send(&data, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    printf("Rank %d sent %f\n", rank, data);

  } else if (rank == 1) {
    // Receive with rank 1
    double data = 0.0;
    MPI_Request request;
    MPI_Irecv(&data, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request);

    printf("Rank %d data before wait %f\n", rank, data);

    MPI_Status status;
    MPI_Wait(&request, &status);
    printf("Rank %d received %f\n", rank, data);

  }

  MPI_Finalize();
}
