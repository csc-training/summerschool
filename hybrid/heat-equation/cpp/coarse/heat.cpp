#include "heat.hpp"
#include "matrix.hpp"
#include <iostream>

void Field::setup(int nx_in, int ny_in) 
{
    nx_full = nx_in;
    ny_full = ny_in;

    nx = nx_full;
    ny = ny_full;

   // matrix includes also ghost layers
   temperature = Matrix<double> (nx + 2, ny + 2);
}

void Field::generate() {

    // Radius of the source disc 
    auto radius = nx_full / 6.0;
    # pragma omp for //shared(nx,ny,radius,temperature) private(dx,dy)
    {
        for (int i = 0; i < nx + 2; i++) {
            for (int j = 0; j < ny + 2; j++) {
                // Distance of point i, j from the origin 
                auto dx = i - nx / 2 + 1;  // Defined inside the parallel region and is thus private to each thread.
                auto dy = j - ny / 2 + 1;
                if (dx * dx + dy * dy < radius * radius) {
                    temperature(i, j) = 5.0;
                } else {
                    temperature(i, j) = 65.0;
                }
            }
        }
    }


    // Boundary conditions
    #pragma omp for
    for (int i = 0; i < nx + 2; i++) {
        // Left
        temperature(i, 0) = 20.0;
        // Right
        temperature(i, ny + 1) = 70.0;
    }
    
    #pragma omp for
    for (int j = 0; j < ny + 2; j++) {
        // Top
        temperature(0, j) = 85.0;
        // Bottom
        temperature(nx + 1, j) = 5.0;
        }    
}
