#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <omp.h>


int main(int argc, char *argv[])
{
    int provided, rank, ntasks;
    int tid, nthreads;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    /* Check that the MPI implementation supports MPI_THREAD_MULTIPLE */
    if (provided < MPI_THREAD_MULTIPLE) {
        printf("MPI does not support MPI_THREAD_MULTIPLE\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
        return 0;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

#pragma omp parallel private(tid, nthreads)
{
    int msg = -1;
    nthreads = omp_get_num_threads();
    tid = omp_get_thread_num();

    if (rank == 0) {
        msg = tid;

#pragma omp single
        {
            printf("%d threads in master rank\n", nthreads);
        }
    }

    // Call broadcast with threads in random order (once per thread)
    MPI_Bcast(&msg, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank > 0) {
        printf("Rank %d thread %d received %d\n", rank, tid, msg);
    }
}

    MPI_Finalize();
    return 0;
}
