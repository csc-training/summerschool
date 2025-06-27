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
    MPI_Comm thread_comm;

    // Create communicators connecting same thread ids across all processes
#pragma omp for ordered schedule(static, 1)
    for (int i = 0; i < nthreads; i++) {
#pragma omp ordered
        {
            MPI_Comm_dup(MPI_COMM_WORLD, &thread_comm);
        }
    }

    if (rank == 0) {
        msg = tid;

#pragma omp single
        {
            printf("%d threads in master rank\n", nthreads);
        }
    }

    // Broadcast using thread-specific communicators
    MPI_Bcast(&msg, 1, MPI_INT, 0, thread_comm);

    if (rank > 0) {
        printf("Rank %d thread %d received %d\n", rank, tid, msg);
    }
}

    MPI_Finalize();
    return 0;
}
