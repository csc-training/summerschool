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
    nthreads = omp_get_num_threads();
    tid = omp_get_thread_num();

    if (rank == 0) {
#pragma omp single
        {
            printf("%d threads in master rank\n", nthreads);
        }
        for (int i = 1; i < ntasks; i++) {
            MPI_Send(&tid, 1, MPI_INT, i, tid, MPI_COMM_WORLD);
        }
    } else {
        int msg;
        MPI_Recv(&msg, 1, MPI_INT, 0, tid, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Rank %d thread %d received %d\n", rank, tid, msg);
    }
}

    MPI_Finalize();
    return 0;
}
