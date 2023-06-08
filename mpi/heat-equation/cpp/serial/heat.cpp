#include "heat.hpp"
#include "matrix.hpp"
#include <iostream>
#include <mpi.h>

void Field::setup(int nx_in, int ny_in)
{
    nx = nx_in;
    ny = ny_in;

   // matrix includes also ghost layers
   temperature = Matrix<double> (nx + 2, ny + 2);
}

void Field::generate() {

    // Radius of the source disc
    auto radius = nx / 6.0;
    for (int i = 0; i < nx + 2; i++) {
        for (int j = 0; j < ny + 2; j++) {
            // Distance of point i, j from the origin
            auto dx = i - nx / 2 + 1;
            auto dy = j - ny / 2 + 1;
            if (dx * dx + dy * dy < radius * radius) {
                temperature(i, j) = 5.0;
            } else {
                temperature(i, j) = 65.0;
            }
        }
    }

    // Boundary conditions
    for (int i = 0; i < nx + 2; i++) {
        // Left
        temperature(i, 0) = 20.0;
        // Right
        temperature(i, ny + 1) = 70.0;
    }

    // Top
    for (int j = 0; j < ny + 2; j++) {
        temperature(0, j) = 85.0;
    }
    // Bottom
    for (int j = 0; j < ny + 2; j++) {
        temperature(nx + 1, j) = 5.0;
    }
}
