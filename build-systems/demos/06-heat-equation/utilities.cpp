// Utility functions for heat equation solver
//    NOTE: This file does not need to be edited! 

#include "heat.hpp"

// Calculate average temperature
double average(const Field& field)
{
     double average = 0.0;

     for (int i = 1; i < field.nx + 1; i++) {
       for (int j = 1; j < field.ny + 1; j++) {
         average += field.temperature(i, j);
       }
     }

     average /= (field.nx_full * field.ny_full);
     return average;
}
