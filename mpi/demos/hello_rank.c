#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
  int size, rank, resultlen;
  char node_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the number of processes in the communicator
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the current process
  MPI_Get_processor_name(node_name, &resultlen);

  //if (rank==0){
  //  printf("Hello from rank %d of %d. The size is %d.\n", rank, size, size);
  //}

  printf("Hello from rank %d of %d is on %s.\n", rank, size, node_name);

  MPI_Finalize();
}
