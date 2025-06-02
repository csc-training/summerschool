#include <stdio.h>
#include <mpi.h>

void print_hello()
{
    int ntasks, rank, result_len;
    char hostname[MPI_MAX_PROCESSOR_NAME];

    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(hostname, &result_len);

    printf("Hello world from task %d of %d in host %s\n", rank, ntasks, hostname);
}
