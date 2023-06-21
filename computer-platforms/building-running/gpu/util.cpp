#include <stdio.h>
#include <mpi.h>
#include <hip/hip_runtime.h>

void print_hello()
{
    int node_rank, result_len;
    MPI_Comm intranode_comm;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int num_gpus;

    MPI_Get_processor_name(hostname, &result_len);
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,  MPI_INFO_NULL, &intranode_comm);

    MPI_Comm_rank(intranode_comm, &node_rank);
    hipGetDeviceCount(&num_gpus);

    if (0 == node_rank) {
       printf("Hello from node %s with %d GPUs\n", hostname, num_gpus);
    }
}
