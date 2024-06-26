#include <cstdio>
#include <cmath>
#include <mpi.h>

constexpr int n = 840;

int main(int argc, char** argv)
{
  
  int rank, ntasks;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
   if (ntasks != 2) {
      printf("This example works only with two processes!\n");
      MPI_Finalize();
      return -1;
   }

  MPI_Status status;
  int istart = 1;
  int istop = n;
  int n_divide = n/2;
  double pi = 0.0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank==0){
    printf("Computing approximation to pi with N=%d\n", n);
    printf("Using %d MPI processes\n", ntasks);

    for (int i=istart; i <= n_divide; i++) {
      double x = (i - 0.5) / n;
      pi += 1.0 / (1.0 + x*x);
    }
    MPI_Send(&pi, 1, MPI_DOUBLE, 1, 0,  MPI_COMM_WORLD);
  }
  else if (rank==1){
    double recv_sum;
    for (int i=n_divide+1; i <= istop; i++) {
      double x = (i - 0.5) / n;
      pi += 1.0 / (1.0 + x*x);
    }
    MPI_Recv(&recv_sum, 1, MPI_DOUBLE, 0, 0,  MPI_COMM_WORLD, &status);
    pi += recv_sum;
    pi *= 4.0 / n;
    printf("Approximate pi=%18.16f (exact pi=%10.8f)\n", pi, M_PI);
  }

  

  

  MPI_Finalize();

}
