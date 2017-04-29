#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>  
#include <omp.h>


int main(int argc, char *argv[])
{
  int provided, rank, ntasks;
  int tid, nthreads, msg, i;
  
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

  /* Check that the MPI implementation supports MPI_THREAD_MULTIPLE */
  if (provided < MPI_THREAD_MULTIPLE) {
    printf("MPI does not support MPI_THREAD_MULTIPLE\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
    return 0;
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

#pragma omp parallel private(msg, tid, nthreads, i)
{
  nthreads = omp_get_num_threads();
  tid = omp_get_thread_num();
    
  if (rank == 0) {
#pragma omp single
{
  printf("%i threads in master rank\n", nthreads);
}
 for (i = 1; i < ntasks; i++)
   MPI_Send(&tid, 1, MPI_INTEGER, i, tid, MPI_COMM_WORLD);
  } else {
    MPI_Recv(&msg, 1, MPI_INTEGER, 0, tid, MPI_COMM_WORLD,
	     MPI_STATUS_IGNORE);
  printf("Rank %i thread %i received %i\n", rank, tid, msg);
 }
}

  MPI_Finalize();
  return 0;
}      
