#include <stdio.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char *argv[])
{
    int my_id, omp_rank;
    int provided, required=MPI_THREAD_FUNNELED;

    /* TODO: Initialize MPI with thread support. */

    /* TODO: Find out the MPI rank and thread ID of each thread and print
     *       out the results. */

    /* TODO: Investigate the provided thread support level. */

    MPI_Finalize();
    return 0;
}
