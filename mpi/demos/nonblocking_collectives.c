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
  fflush(stdout);

  double t0 = MPI_Wtime();

  // Computing 1
  usleep(1000 * rank);
  double data1 = 1.0 * rank;

  // Allreduce
  double reduced;
#ifdef NONBLOCKING
  MPI_Request request;
  MPI_Iallreduce(&data1, &reduced, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request);
#else
  MPI_Allreduce(&data1, &reduced, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  // Computing 2 (independent of 1)
  usleep(1000 * (size - rank));
  double data2 = 2.0 * rank;

#ifdef NONBLOCKING
  MPI_Wait(&request, MPI_STATUS_IGNORE);
#endif

  // Computing 3 (requiring 1 and 2)
  usleep(1000);
  double data3 = reduced + data2;

  double t1 = MPI_Wtime();

  printf("Rank %d computing took %.2f ms, data: %6.1f, %6.1f, %6.1f\n", rank, (t1-t0) * 1e3, data1, data2, data3);

  MPI_Finalize();
}
