#include <cstdio>
#include <mpi.h>
#include <omp.h>

int main(int argc, char *argv[])
{
    int my_id, omp_rank;
    int provided;
    int required = MPI_THREAD_SINGLE;
    //int required = MPI_THREAD_FUNNELED;
    //int required = MPI_THREAD_SERIALIZED;
    //int required = MPI_THREAD_MULTIPLE;

    /* TODO: Initialize MPI with thread support. */
    MPI_Init_thread(&argc, &argv, required, &provided);

    /* TODO: Find out the MPI rank and thread ID of each thread and print
     *       out the results. */
    MPI_Comm_rank(MPI_COMM_WORLD, &omp_rank);

    # pragma omp parallel shared(my_id, omp_rank)
    {
    my_id = omp_get_thread_num();
    printf("Thread rank: %d. Thread id: %d.\n", omp_rank, my_id);  // Will not see any difference here, even with different required, as we will only see the difference when using MPI calls.
    }

    /* TODO: Investigate the provided thread support level. */
    int got_required = provided == required;
    printf("Required thread support level provided (0=no, 1=yes): %d. Thread support level: %d\n", got_required, provided); 

    MPI_Finalize();
    return 0;
}
