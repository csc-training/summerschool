#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
  int size, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  printf("Hello from rank %d of %d\n", rank, size);

  double data[2] = {1. * rank, -1. * rank};
  printf("Rank %d data is                     (%.1f, %.1f)\n", rank, data[0], data[1]);

  double sum_of_data[2];
  MPI_Reduce(data, sum_of_data, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
//  MPI_Allreduce(data, sum_of_data, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//  MPI_Allreduce(MPI_IN_PLACE, data, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  printf("Rank %d data after reduce is        (%.1f, %.1f)\n", rank, data[0], data[1]);
  printf("Rank %d sum_of_data after reduce is (%.1f, %.1f)\n", rank, sum_of_data[0], sum_of_data[1]);

  MPI_Finalize();
}
