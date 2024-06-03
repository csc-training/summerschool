#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
  int size, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  printf("Hello from rank %d of %d\n", rank, size);

  MPI_Barrier(MPI_COMM_WORLD);

  printf("Hello again from rank %d\n", rank);

  MPI_Finalize();
}
