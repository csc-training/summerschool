// Main solver routines for heat equation solver

#include <mpi.h>

#include "heat.hpp"

// Exchange the boundary values, MPI_Sendrecv version
void exchange(Field& field, const ParallelData parallel)
{

    double* sbuf;
    double* rbuf;
    //int rank;
    //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

    // TODO start: implement halo exchange

    // You can utilize the data() method of the Matrix class to obtain pointer
    // to element, e.g. field.temperature.data(i, j)

    // Get the ranks of the up and down neighbours. Grid boundaries treated with MPI_PROC_NULL.
    int up_rank = parallel.nup;
    int down_rank = parallel.ndown;

    // Send to up, receive from down
    sbuf = field.temperature.data(1, 0);  // Send field.temperature.data(1, 0) because field.temperature.data(0, 0) is (the address of) a ghost layer that is received from up_rank.
    rbuf = field.temperature.data(field.nx + 1, 0);  // field.temperature.data(field.nx + 1, 0) is the ghost layer of the current rank used to receive data from down_rank.
    MPI_Sendrecv(sbuf, field.ny+2, MPI_DOUBLE, up_rank, 1, 
    rbuf, field.ny+2, MPI_DOUBLE, down_rank, 1, MPI_COMM_WORLD, &status);  // The field also has ghost layers to the left and right determined by boundary conditions. The total length is therefore field.temperature.data.ny+2.

    // Send to down, receive from up
    sbuf = field.temperature.data(field.nx, 0);  // field.temperature.data(field.nx, 0) is the ghost layer of down_rank.
    rbuf = field.temperature.data(0, 0); // Layer 0 is a ghost layer for receiving data from up_rank.
    MPI_Sendrecv(sbuf, field.ny+2, MPI_DOUBLE, down_rank, 2, 
    rbuf, field.ny+2, MPI_DOUBLE, up_rank, 2, MPI_COMM_WORLD, &status);  // The field also has ghost layers to the left and right determined by boundary conditions. The total length is therefore field.temperature.data.ny+2.


    // TODO end
}


// Update the temperature values using five-point stencil */
void evolve(Field& curr, const Field& prev, const double a, const double dt)
{

  // Compilers do not necessarily optimize division to multiplication, so make it explicit
  auto inv_dx2 = 1.0 / (prev.dx * prev.dx);
  auto inv_dy2 = 1.0 / (prev.dy * prev.dy);

  // Determine the temperature field at next time step
  // As we have fixed boundary conditions, the outermost gridpoints
  // are not updated.
  for (int i = 1; i < curr.nx + 1; i++) {
    for (int j = 1; j < curr.ny + 1; j++) {
            curr(i, j) = prev(i, j) + a * dt * (
	       ( prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j) ) * inv_dx2 +
	       ( prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1) ) * inv_dy2
               );
    }
  }

}


// Exchange the boundary values, normal version, non-blocking version
void exchange_evolve_non_block(Field& curr, Field& prev, const double a, const double dt, const ParallelData parallel)
{

    double* sbuf;
    double* rbuf;

    // TODO start: implement halo exchange

    // You can utilize the data() method of the Matrix class to obtain pointer
    // to element, e.g. field.temperature.data(i, j)

    // Get the ranks of the up and down neighbours. Grid boundaries treated with MPI_PROC_NULL.
    int up_rank = parallel.nup;
    int down_rank = parallel.ndown;

    //MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
    //    int source, int tag, MPI_Comm comm, MPI_Request *request)

    MPI_Status statuses[4];
    MPI_Request requests[4];

    // Send to up, receive from down
    sbuf = prev.temperature.data(1, 0);  // Send prev.temperature.data(1, 0) because prev.temperature.data(0, 0) is (the address of) a ghost layer that is received from up_rank.
    rbuf = prev.temperature.data(prev.nx + 1, 0);  // prev.temperature.data(prev.nx + 1, 0) is the ghost layer of the current rank used to receive data from down_rank.
    MPI_Irecv(rbuf, prev.ny+2, MPI_DOUBLE, down_rank, 1, parallel.cart_comm, &requests[0]);
    MPI_Isend(sbuf, prev.ny+2, MPI_DOUBLE, up_rank, 1, parallel.cart_comm, &requests[1]);

    // Send to down, receive from up
    sbuf = prev.temperature.data(prev.nx, 0);  // prev.temperature.data(prev.nx, 0) is the ghost layer of down_rank.
    rbuf = prev.temperature.data(0, 0); // Layer 0 is a ghost layer for receiving data from up_rank.
    MPI_Irecv(rbuf, prev.ny+2, MPI_DOUBLE, up_rank, 2, parallel.cart_comm, &requests[2]);
    MPI_Isend(sbuf, prev.ny+2, MPI_DOUBLE, down_rank, 2, parallel.cart_comm, &requests[3]);

    // Compute the inner values of the temperature field (those that do not depend on the ghost layers)
    evolve_inner(curr, prev, a, dt);

    // Wait for MPI_Irecv and MPI_Isend to complete by using requests. Also returns metadata for the communication stored in statuses.
    MPI_Waitall(4, requests, statuses);

    // Compute the ghost layer temperature field
    evolve_ghost_boundary(curr, prev, a, dt);

    // TODO end
}


// Update the inner values of the temperature field (those that do not depend on the ghost layers) using five-point stencil */
void evolve_inner(Field& curr, Field& prev, const double a, const double dt)
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


// Update the ghost bondary values of the temperature field (those that do not depend on the ghost layers) using five-point stencil */
void evolve_ghost_boundary(Field& curr, Field& prev, const double a, const double dt)
{

  // Compilers do not necessarily optimize division to multiplication, so make it explicit
  auto inv_dx2 = 1.0 / (prev.dx * prev.dx);
  auto inv_dy2 = 1.0 / (prev.dy * prev.dy);

  // Determine the temperature field at next time step
  // As we have fixed boundary conditions, the outermost gridpoints
  // are not updated.
  for (int j = 1; j < curr.ny + 1; j++) {
    int i = 1;  // Update the upper ghost boundary
    curr(i, j) = prev(i, j) + a * dt * (
	       ( prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j) ) * inv_dx2 +
	       ( prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1) ) * inv_dy2
               );
  }

  for (int j = 1; j < curr.ny + 1; j++) {
    int i = curr.nx;  // Update the lower ghost boundary
    curr(i, j) = prev(i, j) + a * dt * (
	       ( prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j) ) * inv_dx2 +
	       ( prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1) ) * inv_dy2
               );
    
  }

}
