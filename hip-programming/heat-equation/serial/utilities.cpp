/* Utility functions for heat equation solver
 *   NOTE: This file does not need to be edited! */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "heat.h"


/* Copy data on temperature1 into temperature2 */
void copy_field(field *temperature1, field *temperature2)
{
    assert(temperature1->nx == temperature2->nx);
    assert(temperature1->ny == temperature2->ny);
    memcpy(temperature2->data, temperature1->data,
           (temperature1->nx + 2) * (temperature1->ny + 2) * sizeof(double));
}

/* Swap the data of fields temperature1 and temperature2 */
void swap_fields(field *temperature1, field *temperature2)
{
    double *tmp;
    tmp = temperature1->data;
    temperature1->data = temperature2->data;
    temperature2->data = tmp;
}

/* Allocate memory for a temperature field and initialise it to zero */
void allocate_field(field *temperature)
{
    // Allocate also ghost layers
    temperature->data = new double [(temperature->nx + 2) * (temperature->ny + 2)];

    // Initialize to zero
    memset(temperature->data, 0.0,
           (temperature->nx + 2) * (temperature->ny + 2) * sizeof(double));
}

/* Calculate average temperature */
double average(field *temperature)
{
     double average = 0.0;

     for (int i = 1; i < temperature->nx + 1; i++) {
       for (int j = 1; j < temperature->ny + 1; j++) {
         int ind = i * (temperature->ny + 2) + j;
         average += temperature->data[ind];
       }
     }

     average /= (temperature->nx_full * temperature->ny_full);
     return average;
}


