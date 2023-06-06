// Main solver routines for heat equation solver

#include <mpi.h>

#include "heat.hpp"

// Exchange the boundary values
void exchange_init(Field& field, ParallelData& parallel)
{
    // Send to the up, receive from down
    double* sbuf = field.temperature.data(1, 0);
    double* rbuf  = field.temperature.data(field.nx + 1, 0);
    MPI_Isend(sbuf, 1, parallel.rowtype, parallel.nghbrs[0][0], 11,
              parallel.comm, &parallel.requests[0]);
    MPI_Irecv(rbuf, 1, parallel.rowtype, parallel.nghbrs[0][1], 11,
              parallel.comm, &parallel.requests[1]);

    // Send to the down, receive from up
    sbuf = field.temperature.data(field.nx, 0);
    rbuf = field.temperature.data();
    MPI_Isend(sbuf, 1, parallel.rowtype, parallel.nghbrs[0][1], 12,
              parallel.comm, &parallel.requests[2]);
    MPI_Irecv(rbuf, 1, parallel.rowtype, parallel.nghbrs[0][0], 12,
              parallel.comm, &parallel.requests[3]);

    // Send to the left, receive from right
    sbuf = field.temperature.data(0, 1);
    rbuf  = field.temperature.data(0, field.ny + 1);
    MPI_Isend(sbuf, 1, parallel.columntype, parallel.nghbrs[1][0], 13,
              parallel.comm, &parallel.requests[4]);
    MPI_Irecv(rbuf, 1, parallel.columntype, parallel.nghbrs[1][1], 13,
              parallel.comm, &parallel.requests[5]);

    // Send to the right, receive from left
    sbuf = field.temperature.data(0, field.ny);
    rbuf = field.temperature.data();
    MPI_Isend(sbuf, 1, parallel.columntype, parallel.nghbrs[1][1], 14,
              parallel.comm, &parallel.requests[6]);
    MPI_Irecv(rbuf, 1, parallel.columntype, parallel.nghbrs[1][0], 14,
              parallel.comm, &parallel.requests[7]);
}

void exchange_finalize(ParallelData& parallel)
{
    MPI_Waitall(8, parallel.requests, MPI_STATUSES_IGNORE);
}

// Update the temperature values using five-point stencil */
void evolve_interior(Field& curr, const Field& prev, const double a, const double dt)
{

  // Compilers do not necessarily optimize division to multiplication, so make it explicit
  auto inv_dx2 = 1.0 / (prev.dx * prev.dx);
  auto inv_dy2 = 1.0 / (prev.dy * prev.dy);

  // Determine the temperature field at next time step
  // As we have fixed boundary conditions, the outermost gridpoints
  // are not updated.
  for (int i = 2; i < curr.nx; i++) {
    for (int j = 2; j < curr.ny; j++) {
            curr(i, j) = prev(i, j) + a * dt * (
	       ( prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j) ) * inv_dx2 +
	       ( prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1) ) * inv_dy2
               );
    }
  }

}

void evolve_edges(Field& curr, const Field& prev, const double a, const double dt)
{

  // Compilers do not necessarily optimize division to multiplication, so make it explicit
  auto inv_dx2 = 1.0 / (prev.dx * prev.dx);
  auto inv_dy2 = 1.0 / (prev.dy * prev.dy);

  // Determine the temperature field at next time step
  // As we have fixed boundary conditions, the outermost gridpoints
  // are not updated.
  // Boundaries in first dimension
  for (int i = 1; i < curr.nx + 1; i += curr.nx - 1) {
    for (int j = 1; j < curr.ny + 1; j++) {
            curr(i, j) = prev(i, j) + a * dt * (
	       ( prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j) ) * inv_dx2 +
	       ( prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1) ) * inv_dy2
               );
    }
  }

  // Boundaries in seconddimension
  for (int i = 1; i < curr.nx + 1; i++) {
    for (int j = 1; j < curr.ny + 1; j += curr.ny - 1) {
            curr(i, j) = prev(i, j) + a * dt * (
	       ( prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j) ) * inv_dx2 +
	       ( prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1) ) * inv_dy2
               );
    }
  }

}
