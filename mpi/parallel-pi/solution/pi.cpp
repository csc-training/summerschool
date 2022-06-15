#include <cstdio>
#include <cmath>
#include <mpi.h>

constexpr int n = 840;

int main(int argc, char** argv)
{
   int myid, ntasks;

   MPI_Status status;

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   if (0 == myid) {
      printf("Computing approximation to pi with N=%d\n", n);
      printf("Using %d MPI processes\n", ntasks);
   }

   int chunksize = n / ntasks;
   int istart = myid * chunksize + 1;
   int istop = (myid + 1) * chunksize;

   // Handle possible uneven division
   int remainder = n % ntasks;
   if (remainder > 0) {
       if (myid < remainder) {
          // Assign this task one element more
          istart += myid;
          istop += myid + 1;
       } else {
          istart += remainder;
          istop += remainder;
       }
    }
         
   double localpi = 0.0;
   for (int i=istart; i <= istop; i++) {
     double x = (i - 0.5) / n;
     localpi += 1.0 / (1.0 + x*x);
   }

   // Reduction to rank 0
   if (0 == myid) {
      double pi = localpi;
      for (int i=1; i < ntasks; i++) {
        MPI_Recv(&localpi, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        pi += localpi;
      }
      pi *= 4.0 / n;
      printf("Approximate pi=%18.16f (exact pi=%10.8f)\n", pi, M_PI);
   } else {
      MPI_Send(&localpi, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
   }

   MPI_Finalize();

}
   

