#include "heat.hpp"
#include "matrix.hpp"
#include <iostream>
#include <mpi.h>

void Field::setup(int nx_in, int ny_in, ParallelData &parallel) 
{
  nx_full = nx_in;
  ny_full = ny_in;

  int dims[2], periods[2], coords[2];
  MPI_Cart_get(parallel.comm, 2, dims, periods, coords);
  
  nx = nx_full / dims[0];
  if (nx * dims[0] != nx_full) {
    std::cout << "Cannot divide grid evenly to processors" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, -2);
  }
  ny = ny_full / dims[1];
  if (ny * dims[1] != ny_full) {
    std::cout << "Cannot divide grid evenly to processors" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, -2);
  }
  
  // matrix includes also ghost layers
  temperature = Matrix<double> (nx + 2, ny + 2);
  
  // MPI datatypes for halo exchange
  MPI_Type_vector(nx + 2, 1, ny + 2, MPI_DOUBLE,
                  &parallel.columntype);
  MPI_Type_contiguous(ny + 2, MPI_DOUBLE, &parallel.rowtype);
  MPI_Type_commit(&parallel.columntype);
  MPI_Type_commit(&parallel.rowtype);

  // MPI datatype for subblock needed in I/O
  // Rank 0 uses datatype for receiving data into full array while
  // other ranks use datatype for sending the inner part of array
    int sizes[2];
    int subsizes[2] = { nx, ny };
    int offsets[2] = { 0, 0 };
    if (parallel.rank == 0) {
        sizes[0] = nx_full;
        sizes[1] = ny_full;
    } else {
        sizes[0] = nx + 2;
        sizes[1] = ny + 2;
    }
    MPI_Type_create_subarray(2, sizes, subsizes, offsets, MPI_ORDER_C,
                             MPI_DOUBLE, &parallel.subarraytype);
    MPI_Type_commit(&parallel.subarraytype);    
}

void Field::generate(ParallelData parallel) {

    int dims[2], coords[2], periods[2];
    MPI_Cart_get(parallel.comm, 2, dims, periods, coords);

    // Radius of the source disc 
    auto radius = nx_full / 6.0;
    for (int i = 0; i < nx + 2; i++) {
        for (int j = 0; j < ny + 2; j++) {
            // Distance of point i, j from the origin 
            auto dx = i + coords[0] * nx - nx_full / 2 + 1;
            auto dy = j + coords[1] * ny - ny_full / 2 + 1;
            if (dx * dx + dy * dy < radius * radius) {
                temperature(i, j) = 5.0;
            } else {
                temperature(i, j) = 65.0;
            }
        }
    }

    // Boundary conditions
    // Left
    if (0 == coords[1]) {
      for (int i = 0; i < nx + 2; i++) {
        temperature(i, 0) = 20.0;
      }
    }
    // Right
    if (coords[1] == dims[1] - 1 ) {
      for (int i = 0; i < nx + 2; i++) {
        temperature(i, ny + 1) = 70.0;
      }
    }

    // Top
    if (0 == coords[0]) {
        for (int j = 0; j < ny + 2; j++) {
            temperature(0, j) = 85.0;
        }
    }
    // Bottom
    if (coords[0] == dims[0] - 1) {
        for (int j = 0; j < ny + 2; j++) {
            temperature(nx + 1, j) = 5.0;
        }
    }
}
