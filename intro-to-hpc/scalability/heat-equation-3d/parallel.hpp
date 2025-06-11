#pragma once
#ifndef NO_MPI
#include <mpi.h>
#endif
#include "matrix.hpp"
#include <hip/hip_runtime.h>

// Class for basic parallelization information
struct ParallelData {
    int size;            // Number of MPI tasks
    int dims[3] = {0, 0, 0};
    int coords[3] = {0, 0, 0};
    int rank;
    int dev_count;
    int ngbrs[3][2];     // Ranks of neighbouring MPI tasks
#ifndef NO_MPI
#if defined MPI_DATATYPES || defined MPI_NEIGHBORHOOD
    MPI_Datatype halotypes[3];
#else
    double* send_buffers[3][2];
    double* recv_buffers[3][2];
#endif
    MPI_Datatype subarraytype;
    MPI_Request requests[12];
    MPI_Comm comm;
#endif

    ParallelData() {     // Constructor

#ifdef NO_MPI
      size = 1;
      rank = 0;
      dims[0] = dims[1] = dims[2] = 1;
#else
      MPI_Comm_size(MPI_COMM_WORLD, &size);

      constexpr int ndims = 3;
      int periods[ndims] = {0, 0, 0};

#ifdef MPI_3D_DECOMPOSITION
      MPI_Dims_create(size, ndims, dims);
#else
      // Non-contiguous memcpys in communication may be expensive so use only 1D here
      dims[0] = size;
      dims[1] = 1;
      dims[2] = 1;
#endif

      MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1, &comm);
      MPI_Comm_rank(comm, &rank);
      MPI_Cart_get(comm, 3, dims, periods, coords);

      // Determine neighbors
      for (int i=0; i < ndims; i++)
        MPI_Cart_shift(comm, i, 1, &ngbrs[i][0], &ngbrs[i][1]);

      // Set GPU device
      MPI_Comm intranode_comm;
      int node_rank, node_procs;

      MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,  MPI_INFO_NULL, &intranode_comm);

      MPI_Comm_rank(intranode_comm, &node_rank);
      MPI_Comm_size(intranode_comm, &node_procs);

      MPI_Comm_free(&intranode_comm);

      GPU_CHECK( hipGetDeviceCount(&dev_count) );

      // Allow oversubscribing devices
      int my_device = node_rank % dev_count;
      if (node_procs > dev_count) {
         if (0 == rank) 
            std::cout << "Oversubscriging GPUs: " <<
                         "MPI tasks per node: " << node_procs <<
                         ", GPUs per node: " << dev_count << std::endl;
      }

      GPU_CHECK( hipSetDevice(my_device) );

#endif
    };
};

