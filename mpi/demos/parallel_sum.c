#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
  int size, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  printf("Hello from rank %d of %d\n", rank, size);

  double buffer[10];

  // Allocate data on rank 0
  if (rank == 0) {
    for (int i = 0; i < 10; ++i) {
      buffer[i] = i;
    }
  }

  // Distribute data
  if (rank == 0) {
    // Send half of data to rank 1
    MPI_Send(&buffer[5], 5, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
  } else if (rank == 1) {
    // Receive half of data from rank 0
    MPI_Recv(buffer, 10, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Compute partial sum in both ranks separately
  double partial_sum = 0;
  for (int i = 0; i < 5; ++i) {
    printf("Rank %d: buff[%d] = %f\n", rank, i, buffer[i]);
    partial_sum += buffer[i];
  }
  printf("Rank %d: partial sum is %f\n", rank, partial_sum);

  // Collect data to rank 0
  double sum;
  if (rank == 0) {
    // Receive rank 1's partial sum
    MPI_Recv(&sum, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Add rank 0's partial sum
    sum += partial_sum;
  } else if (rank == 1) {
    MPI_Send(&partial_sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }

  // Print final sum
  printf("Rank %d: final sum is %f\n", rank, sum);

  MPI_Finalize();
}
