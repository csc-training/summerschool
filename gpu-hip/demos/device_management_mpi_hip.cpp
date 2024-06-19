#include <cstdio>
#include <mpi.h>
#include <hip/hip_runtime.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Create communicator per node
    MPI_Comm comm_node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &comm_node);
    int size_node, rank_node;
    MPI_Comm_size(comm_node, &size_node);
    MPI_Comm_rank(comm_node, &rank_node);

    int namelen;
    char procname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(procname, &namelen);

    int count, device;
    hipGetDeviceCount(&count);
    hipGetDevice(&device);

    printf("I'm MPI rank %2d/%-2d (world) %2d/%-2d (node) on %s with GPU %2d/%-2d\n",
           rank, size, rank_node, size_node, procname, device, count);

    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    hipSetDevice(rank_node % count);
    hipGetDevice(&device);
    printf("Now MPI rank %2d/%-2d (world) %2d/%-2d (node) on %s with GPU %2d/%-2d\n",
           rank, size, rank_node, size_node, procname, device, count);

    MPI_Finalize();

    return 0;
}
