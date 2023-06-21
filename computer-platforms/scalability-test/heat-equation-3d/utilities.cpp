// Utility functions for heat equation solver
//    NOTE: This file does not need to be edited! 

#ifdef NO_MPI
#include <omp.h>
#else
#include <mpi.h>
#endif
#include "heat.hpp"

// Calculate average temperature
double average(const Field& field)
{
     double local_average = 0.0;
     double average = 0.0;

     for (int i = 1; i < field.nx + 1; i++) {
       for (int j = 1; j < field.ny + 1; j++) {
         for (int k = 1; k < field.nz + 1; k++) {
           local_average += field.temperature(i, j, k);
         }
       }
     }

#ifdef NO_MPI
     average = local_average;
#else
     MPI_Allreduce(&local_average, &average, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
     average /= (field.nx_full * field.ny_full * field.nz_full);
     return average;
}

double timer() 
{
    double t0;
#ifdef NO_MPI
    t0 = omp_get_wtime();
#else
    t0 = MPI_Wtime();
#endif
    return t0;
}

