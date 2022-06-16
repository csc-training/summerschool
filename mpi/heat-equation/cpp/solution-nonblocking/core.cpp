// Main solver routines for heat equation solver

#include <mpi.h>

#include "heat.hpp"

// Exchange the boundary values
void exchange_init(Field& field, ParallelData& parallel)
{

    // Send to up, receive from down
    double* sbuf_up = field.temperature.data(1, 0);
    double* rbuf_down  = field.temperature.data(field.nx + 1, 0);
    MPI_Isend(sbuf_up, field.ny + 2, MPI_DOUBLE, 
	      parallel.nup, 11, MPI_COMM_WORLD, &parallel.requests[0]); 
    MPI_Irecv(rbuf_down, field.ny + 2, MPI_DOUBLE, 
	      parallel.ndown, 11, MPI_COMM_WORLD, &parallel.requests[1]); 

    // Send to down, receive from up
    double* sbuf_down = field.temperature.data(field.nx, 0);
    double* rbuf_up = field.temperature.data();
    MPI_Isend(sbuf_down, field.ny + 2, MPI_DOUBLE, 
              parallel.ndown, 12, MPI_COMM_WORLD, &parallel.requests[2]); 
    MPI_Irecv(rbuf_up, field.ny + 2, MPI_DOUBLE,
              parallel.nup, 12, MPI_COMM_WORLD, &parallel.requests[3]);

}

void exchange_finalize(ParallelData& parallel)
{
    MPI_Waitall(4, parallel.requests, MPI_STATUSES_IGNORE);
}

// Update the temperature values using five-point stencil 
// in the border-independent region of the field
void evolve_interior(Field& curr, const Field& prev, const double a, const double dt)
{

  // Compilers do not necessarily optimize division to multiplication, so make it explicit
  auto inv_dx2 = 1.0 / (prev.dx * prev.dx);
  auto inv_dy2 = 1.0 / (prev.dy * prev.dy);

  // Determine the temperature field at next time step
  // As we have fixed boundary conditions, the outermost gridpoints
  // are not updated.
  for (int i = 2; i < curr.nx; i++) {
    for (int j = 1; j < curr.ny + 1; j++) {
            curr(i, j) = prev(i, j) + a * dt * (
	       ( prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j) ) * inv_dx2 +
	       ( prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1) ) * inv_dy2
               );
    }
  }

}

// Update the temperature values using five-point stencil 
// in the border regions of the field
void evolve_edges(Field& curr, const Field& prev, const double a, const double dt)
{

  // Compilers do not necessarily optimize division to multiplication, so make it explicit
  auto inv_dx2 = 1.0 / (prev.dx * prev.dx);
  auto inv_dy2 = 1.0 / (prev.dy * prev.dy);

  // Determine the temperature field at next time step
  // As we have fixed boundary conditions, the outermost gridpoints
  // are not updated.
  for (int j = 1; j < curr.ny + 1; j++) {
      int i = 1;
      curr(i, j) = prev(i, j) + a * dt * (
	       ( prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j) ) * inv_dx2 +
	       ( prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1) ) * inv_dy2
               );
    }

  for (int j = 1; j < curr.ny + 1; j++) {
      int i = curr.nx;
      curr(i, j) = prev(i, j) + a * dt * (
	       ( prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j) ) * inv_dx2 +
	       ( prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1) ) * inv_dy2
               );
    }
}
