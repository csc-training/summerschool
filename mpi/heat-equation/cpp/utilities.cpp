// Utility functions for heat equation solver

#include <mpi.h>

#include "heat.hpp"

// Calculate average temperature
double average(const Field& field, const ParallelData parallel)
{
     double local_average = 0.0;
     double average = 0.0;

     for (int i = 1; i < field.nx + 1; i++) {
       for (int j = 1; j < field.ny + 1; j++) {
         local_average += field.temperature(i, j);
       }
     }


    // Calculate the overall average by getting the local averages from the other processes
    MPI_Reduce(&local_average, &average, 1, MPI_DOUBLE, MPI_SUM, 0, parallel.cart_comm);
               
    average /= (field.nx_full * field.ny_full);

    /*
     if (0 == parallel.rank) {
         average = local_average;
         for (int p=1; p < parallel.size; p++) {
             MPI_Recv(&local_average, 1, MPI_DOUBLE, p, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // Rank 0 receives the summed temperature from all other ranks.
             average += local_average;
         }
         average /= (field.nx_full * field.ny_full);
     } else {
         MPI_Send(&local_average, 1, MPI_DOUBLE, 0, 22, MPI_COMM_WORLD);  // All ranks send the summed temperature to rank 0.
     }
    */

     return average;
}
