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

    // Create thread-specific communicators
    int max_threads = omp_get_max_threads();
    MPI_Comm *mpi_comm_thread = (MPI_Comm*)malloc(max_threads * sizeof(MPI_Comm));
    for (int i = 0; i < max_threads; i++) {
        MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm_thread[i]);
    }

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

    // Broadcast using thread-specific communicators
    MPI_Bcast(&msg, 1, MPI_INT, 0, mpi_comm_thread[tid]);

    if (rank > 0) {
        printf("Rank %d thread %d received %d\n", rank, tid, msg);
    }
}

    free(mpi_comm_thread);
    MPI_Finalize();
    return 0;
}
