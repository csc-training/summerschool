// Main solver routines for heat equation solver

#ifndef NO_MPI
#include <mpi.h>
#endif

#include "heat.hpp"
#include "parallel.hpp"
#include <hip/hip_runtime.h>
#include "error_checks.h"

// Exchange the boundary values
#ifdef MPI_DATATYPES
void exchange_init_datatypes(Field& field, ParallelData& parallel)
{
    double *sbuf, *rbuf;

    // x-direction
    sbuf = field.devdata(1, 0, 0);
    rbuf = field.devdata(field.nx + 1, 0, 0);
    buf_size = (field.ny + 2) * (field.nz + 2);
    MPI_Isend(sbuf, 1, parallel.halotypes[0],
              parallel.ngbrs[0][0], 11, parallel.comm, &parallel.requests[0]);
    MPI_Irecv(rbuf, 1, parallel.halotypes[0],
              parallel.ngbrs[0][1], 11, parallel.comm, &parallel.requests[1]);

    sbuf = field.devdata(field.nx, 0, 0);
    rbuf = field.devdata(0, 0, 0);
    MPI_Isend(sbuf, 1, parallel.halotypes[0],
              parallel.ngbrs[0][1], 12, parallel.comm, &parallel.requests[2]);
    MPI_Irecv(rbuf, 1, parallel.halotypes[0],
              parallel.ngbrs[0][0], 12, parallel.comm, &parallel.requests[3]);
    
    // y-direction
    sbuf = field.devdata(0, 1, 0);
    rbuf = field.devdata(0, field.ny + 1, 0);
    MPI_Isend(sbuf, 1, parallel.halotypes[1],
              parallel.ngbrs[1][0], 21, parallel.comm, &parallel.requests[4]);
    MPI_Irecv(rbuf, 1, parallel.halotypes[1],
              parallel.ngbrs[1][1], 21, parallel.comm, &parallel.requests[5]);
    
    sbuf = field.devdata(0, field.ny, 0);
    rbuf = field.devdata(0, 0, 0);
    MPI_Isend(sbuf, 1, parallel.halotypes[1],
              parallel.ngbrs[1][1], 22, parallel.comm, &parallel.requests[6]);
    MPI_Irecv(rbuf, 1, parallel.halotypes[1],
              parallel.ngbrs[1][0], 22, parallel.comm, &parallel.requests[7]);
  
    // z-direction
    sbuf = field.devdata(0, 0, 1);
    rbuf = field.devdata(0, 0, field.nz + 1);
    MPI_Isend(sbuf, 1, parallel.halotypes[2],
              parallel.ngbrs[2][0], 31, parallel.comm, &parallel.requests[8]);
    MPI_Irecv(rbuf, 1, parallel.halotypes[2],
              parallel.ngbrs[2][1], 31, parallel.comm, &parallel.requests[9]);
    
    sbuf = field.devdata(0, 0, field.nz);
    rbuf = field.devdata(0, 0, 0);
    MPI_Isend(sbuf, 1, parallel.halotypes[2],
              parallel.ngbrs[2][1], 32, parallel.comm, &parallel.requests[10]);
    MPI_Irecv(rbuf, 1, parallel.halotypes[2],
              parallel.ngbrs[2][0], 32, parallel.comm, &parallel.requests[11]);
}
#endif

#ifdef MPI_NEIGHBORHOOD
void exchange_init_neighborhood(Field& field, ParallelData& parallel)
    MPI_Datatype types[6] = {parallel.halotypes[0], parallel.halotypes[0],
                             parallel.halotypes[1], parallel.halotypes[1],
                             parallel.halotypes[2], parallel.halotypes[2]};
    int counts[6] = {1, 1, 1, 1, 1, 1};
    MPI_Aint sdisps[6], rdisps[6], disp0;

    // Determine displacements
    disp0 = reinterpret_cast<MPI_Aint> (field.devdata());
    sdisps[0] =  reinterpret_cast<MPI_Aint> (field.devdata(1, 0, 0));        
    sdisps[1] =  reinterpret_cast<MPI_Aint> (field.devdata(field.nx, 0, 0)); 
    sdisps[2] =  reinterpret_cast<MPI_Aint> (field.devdata(0, 1, 0));        
    sdisps[3] =  reinterpret_cast<MPI_Aint> (field.devdata(0, field.ny, 0)); 
    sdisps[4] =  reinterpret_cast<MPI_Aint> (field.devdata(0, 0, 1));        
    sdisps[5] =  reinterpret_cast<MPI_Aint> (field.devdata(0, 0, field.nz)); 

    rdisps[0] =  reinterpret_cast<MPI_Aint> (field.devdata(0, 0, 0));            
    rdisps[1] =  reinterpret_cast<MPI_Aint> (field.devdata(field.nx + 1, 0, 0)); 
    rdisps[2] =  reinterpret_cast<MPI_Aint> (field.devdata(0, 0, 0));            
    rdisps[3] =  reinterpret_cast<MPI_Aint> (field.devdata(0, field.ny + 1, 0)); 
    rdisps[4] =  reinterpret_cast<MPI_Aint> (field.devdata(0, 0, 0));            
    rdisps[5] =  reinterpret_cast<MPI_Aint> (field.devdata(0, 0, field.nz + 1)); 

    for (int i=0; i < 6; i++) {
      sdisps[i] -= disp0;
      rdisps[i] -= disp0;
    }

    MPI_Neighbor_alltoallw(field.devdata(), counts, sdisps, types,
                           field.devdata(), counts, rdisps, types,
                           parallel.comm);
}
#endif

#if !(defined MPI_DATATYPES || defined MPI_NEIGHBORHOOD)
// Communicate with manual packing to/from send/receive buffers
void exchange_init_packing(Field& field, ParallelData& parallel)
{
    size_t buf_size;
    double *sbuf, *rbuf;

    // x-direction
    buf_size = (field.ny + 2) * (field.nz + 2);
    // In x-direction buffer is contiguous, so no memory copies are needed

    sbuf = field.devdata(1, 0, 0);
    rbuf = field.devdata(0, 0, 0);

    MPI_Isend(sbuf, buf_size, MPI_DOUBLE,
              parallel.ngbrs[0][0], 11, parallel.comm, &parallel.requests[0]);
    MPI_Irecv(rbuf, buf_size, MPI_DOUBLE,
              parallel.ngbrs[0][0], 11, parallel.comm, &parallel.requests[1]);

    sbuf = field.devdata(field.nx, 0, 0);
    rbuf = field.devdata(field.nx + 1, 0, 0);

    MPI_Isend(sbuf, buf_size, MPI_DOUBLE,
              parallel.ngbrs[0][1], 11, parallel.comm, &parallel.requests[2]);
    MPI_Irecv(rbuf, buf_size, MPI_DOUBLE,
              parallel.ngbrs[0][1], 11, parallel.comm, &parallel.requests[3]);


    // y-direction
    buf_size = (field.nx + 2) * (field.nz + 2);
    // copy to halo
    if (parallel.ngbrs[1][0] != MPI_PROC_NULL) {
      for (int i=0; i < field.nx + 2; i++) {
          GPU_CHECK( hipMemcpy(parallel.send_buffers[1][0] + i * (field.nz + 2), field.devdata(1, 1, 0), (field.nz + 2) *  sizeof(double), hipMemcpyDeviceToDevice) );
      }
    }

    sbuf = parallel.send_buffers[0][0];
    rbuf = parallel.recv_buffers[0][0];

    MPI_Isend(sbuf, buf_size, MPI_DOUBLE,
              parallel.ngbrs[1][0], 11, parallel.comm, &parallel.requests[4]);
    MPI_Irecv(rbuf, buf_size, MPI_DOUBLE,
              parallel.ngbrs[1][0], 11, parallel.comm, &parallel.requests[5]);

    if (parallel.ngbrs[1][1] != MPI_PROC_NULL) {
      for (int i=0; i < field.nx + 2; i++) {
        GPU_CHECK( hipMemcpy(parallel.send_buffers[1][1] + i * (field.nz + 2), field.devdata(1, field.ny, 0), (field.nz + 2) *  sizeof(double), hipMemcpyDeviceToDevice) );
      }
    }

    sbuf = parallel.send_buffers[1][1];
    rbuf = parallel.recv_buffers[1][1];

    MPI_Isend(sbuf, buf_size, MPI_DOUBLE,
              parallel.ngbrs[1][1], 11, parallel.comm, &parallel.requests[6]);
    MPI_Irecv(rbuf, buf_size, MPI_DOUBLE,
              parallel.ngbrs[1][1], 11, parallel.comm, &parallel.requests[7]);

    // z-direction
    buf_size = (field.nx + 2) * (field.ny + 2);
    // copy to halo
    if (parallel.ngbrs[2][0] != MPI_PROC_NULL) {
      for (int i=0; i < field.nx + 2; i++) 
        for (int j=0; j < field.ny + 2; j++) {
           GPU_CHECK( hipMemcpy(parallel.send_buffers[2][0] + j + i * (field.nz + 2), field.devdata(i, j, 1), 
                    sizeof(double), hipMemcpyDeviceToDevice) );
        }
    }

    sbuf = parallel.send_buffers[2][0];
    rbuf = parallel.recv_buffers[2][0];

    MPI_Isend(sbuf, buf_size, MPI_DOUBLE,
              parallel.ngbrs[2][0], 11, parallel.comm, &parallel.requests[8]);
    MPI_Irecv(rbuf, buf_size, MPI_DOUBLE,
              parallel.ngbrs[2][0], 11, parallel.comm, &parallel.requests[9]);

    if (parallel.ngbrs[2][1] != MPI_PROC_NULL) {
      for (int i=0; i < field.nx + 2; i++) 
        for (int j=0; j < field.ny + 2; j++) {
         GPU_CHECK( hipMemcpy(parallel.send_buffers[2][1] + j + i * (field.nz + 2), field.devdata(i, j, field.nz), 
                    sizeof(double), hipMemcpyDeviceToDevice) );
      }
    }

    sbuf = parallel.send_buffers[2][1];
    rbuf = parallel.recv_buffers[2][1];

    MPI_Isend(sbuf, buf_size, MPI_DOUBLE,
              parallel.ngbrs[2][1], 11, parallel.comm, &parallel.requests[10]);
    MPI_Irecv(rbuf, buf_size, MPI_DOUBLE,
              parallel.ngbrs[2][1], 11, parallel.comm, &parallel.requests[11]);
}
#endif

void exchange_init(Field& field, ParallelData& parallel)
{
#ifdef NO_MPI
    return;
#elif defined MPI_DATATYPES
    exchange_init_datatypes(field, parallel);
#elif defined MPI_NEIGHBORHOOD
    exchange_init_neighborhood(field, parallel);
#else
    exchange_init_packing(field, parallel);
#endif
}


void exchange_finalize(Field& field, ParallelData& parallel)
{

#if defined NO_MPI || defined MPI_NEIGHBORHOOD
    return;
#elif defined MPI_DATATYPES
    MPI_Waitall(12, parallel.requests, MPI_STATUSES_IGNORE);
#else
    MPI_Waitall(12, parallel.requests, MPI_STATUSES_IGNORE);

    // copy from halos
    // x-direction
    // In x-direction buffer is contiguous, so no memory copies are needed
    
    // y-direction
    if (parallel.ngbrs[1][0] != MPI_PROC_NULL)
      for (int i=0; i < field.nx + 2; i++)
         GPU_CHECK( hipMemcpy(field.devdata(i, 0, 0), parallel.recv_buffers[1][0],
                   (field.nz + 2) * sizeof(double), hipMemcpyDeviceToDevice) );
    if (parallel.ngbrs[1][1] != MPI_PROC_NULL)
      for (int i=0; i < field.nx + 2; i++)
         GPU_CHECK( hipMemcpy(field.devdata(i, field.ny + 1, 0), parallel.recv_buffers[1][1],
                   (field.nz + 2) * sizeof(double), hipMemcpyDeviceToDevice) );

    // z-direction
    if (parallel.ngbrs[2][0] != MPI_PROC_NULL)
      for (int i=0; i < field.nx + 2; i++)
        for (int j=0; j < field.ny + 2; j++) {
         GPU_CHECK( hipMemcpy(field.devdata(i, j, 0), parallel.recv_buffers[2][0],
                    sizeof(double), hipMemcpyDeviceToDevice) );
        }
    if (parallel.ngbrs[2][1] != MPI_PROC_NULL)
      for (int i=0; i < field.nx + 2; i++)
        for (int j=0; j < field.ny + 2; j++) {
         GPU_CHECK( hipMemcpy(field.devdata(i, j, field.nz + 1), parallel.recv_buffers[2][1],
                    sizeof(double), hipMemcpyDeviceToDevice) );
        }     
#endif
}




