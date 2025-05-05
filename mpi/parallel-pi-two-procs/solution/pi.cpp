#include <cstdio>
#include <cmath>
#include <mpi.h>

constexpr int n = 840;

int main(int argc, char** argv) {

   int rank, ntasks;

   MPI_Status status;

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
   if (ntasks != 2) {
      printf("This example works only with two processes!\n");
      MPI_Finalize();
      return -1;
   }

   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   if (0 == rank) {
      printf("Computing approximation to pi with N=%d\n", n);
      printf("Using %d MPI processes\n", ntasks);
   }

   if (0 == rank) {
      int istart = 1;
      int istop = n / 2;

      double localpi = 0.0;
      for (int i=istart; i <= istop; i++) {
        double x = (i - 0.5) / n;
        localpi += 1.0 / (1.0 + x*x);
      }

      double pi = localpi;
      MPI_Recv(&localpi, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);
      pi += localpi;
      pi *= 4.0 / n;
      printf("Approximate pi=%18.16f (exact pi=%10.8f)\n", pi, M_PI);
   } else if (1 == rank) {
      int istart = n / 2 + 1;
      int istop = n;

      double localpi = 0.0;
      for (int i=istart; i <= istop; i++) {
        double x = (i - 0.5) / n;
        localpi += 1.0 / (1.0 + x*x);
      }

      MPI_Send(&localpi, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
   }

   MPI_Finalize();
}
