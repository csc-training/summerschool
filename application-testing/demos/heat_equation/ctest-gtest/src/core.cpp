// Main solver routines for heat equation solver

#include <mpi.h>

#include "constants.hpp"
#include "core.hpp"
#include "field.hpp"
#include "parallel.hpp"

namespace heat {
// Exchange the boundary values
void exchange(Field &field, const ParallelData &parallel) {
    const auto n = field.num_to_exchange();
    // Send to up, receive from down
    constexpr int tag1 = 11;
    MPI_Sendrecv(field.to_up(), n, MPI_DOUBLE, parallel.nup, tag1,
                 field.from_down(), n, MPI_DOUBLE, parallel.ndown, tag1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send to down, receive from up
    constexpr int tag2 = 12;
    MPI_Sendrecv(field.to_down(), n, MPI_DOUBLE, parallel.ndown, tag2,
                 field.from_up(), n, MPI_DOUBLE, parallel.nup, tag2,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// Update the temperature values using five-point stencil */
void evolve(Field &curr, const Field &prev, const Constants &constants) {
    // Determine the temperature field at next time step
    // As we have fixed boundary conditions, the outermost gridpoints
    // are not updated.
    for (int i = 0; i < curr.num_rows; i++) {
        for (int j = 0; j < curr.num_cols; j++) {
            curr(i, j) = prev.sample(i, j, constants);
        }
    }
}
} // namespace heat
