#include "parallel.hpp"
#include <mpi.h>

namespace heat {
ParallelData::ParallelData() {
    // Get the number of processes and the rank of this process
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // The processes are stacked vertically, previous process is above this,
    // next is below
    nup = rank - 1;
    ndown = rank + 1;

    // The first doesn't have a process above it, the last below it
    if (nup < 0) {
        nup = MPI_PROC_NULL;
    }
    if (ndown > size - 1) {
        ndown = MPI_PROC_NULL;
    }
}
} // namespace heat
