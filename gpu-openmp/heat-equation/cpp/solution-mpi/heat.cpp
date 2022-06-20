#include "heat.hpp"
#include <iostream>
#include <mpi.h>
#include <omp.h>

void Field::setup(int nx_in, int ny_in, ParallelData parallel) 
{
    nx_full = nx_in;
    ny_full = ny_in;

    nx = nx_full / parallel.size;
    if (nx * parallel.size != nx_full) {
        std::cout << "Cannot divide grid evenly to processors" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -2);
    }
    ny = ny_full;

   // size includes ghost layers
   std::size_t field_size = (nx + 2) * (ny + 2);

   temperature = std::vector<double> (field_size);

#ifdef _OPENMP
    MPI_Comm intranodecomm;
    int nodeRank, nodeProcs, devCount;

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,  MPI_INFO_NULL, &intranodecomm);

    MPI_Comm_rank(intranodecomm, &nodeRank);
    MPI_Comm_size(intranodecomm, &nodeProcs);

    MPI_Comm_free(&intranodecomm);

    devCount = omp_get_num_devices();

    if (nodeProcs > devCount) {
        printf("Not enough GPUs for all processes in the node.\n");
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, -2);
    }

    omp_set_default_device(nodeRank);
#endif
   

}

void Field::generate(ParallelData parallel) {

    // Radius of the source disc 
    auto radius = nx_full / 6.0;
    for (int i = 0; i < nx + 2; i++) {
        for (int j = 0; j < ny + 2; j++) {
            // Distance of point i, j from the origin 
            int ind = i * (ny + 2) + j;
            auto dx = i + parallel.rank * nx - nx_full / 2 + 1;
            auto dy = j - ny / 2 + 1;

            if (dx * dx + dy * dy < radius * radius) {
                temperature[ind] = 5.0;
            } else {
                temperature[ind] = 65.0;
            }
        }
    }

    // Boundary conditions
    for (int i = 0; i < nx + 2; i++) {
        // Left
        temperature[i * (ny + 2)] = 20.0;
        // Right
        temperature[i * (ny + 2) + ny + 1] = 70.0;
    }

    // Top
    if (0 == parallel.rank) {
        for (int j = 0; j < ny + 2; j++) {
            int ind = j;
            temperature[ind] = 85.0;            
        }
    }
    // Bottom
    if (parallel.rank == parallel.size - 1) {
        for (int j = 0; j < ny + 2; j++) {
            int ind = (nx + 1) * (ny + 2) + j;
            temperature[ind] = 5.0;
        }
    }
}
