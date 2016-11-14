#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <mpi.h>

/* This macro can be used to check the return code of
   MPI function calls */
#ifndef NDEBUG
#define MPI_CHECK(errcode)                                              \
    if(errcode != MPI_SUCCESS) {                                        \
        fprintf(stderr, "MPI error in %s at line %i\n",                 \
                __FILE__, __LINE__);                                    \
        MPI_Abort(MPI_COMM_WORLD, errcode);                             \
        MPI_Finalize();                                                 \
        exit(errcode);                                                  \
    }
#endif


int main(int argc, char *argv[])
{
    int provided, rank, ntasks;
    int tid, nthreads, msg, i;

    MPI_CHECK(MPI_Init_thread
              (&argc, &argv, MPI_THREAD_MULTIPLE, &provided));

    /* Check that the MPI implementation supports MPI_THREAD_MULTIPLE */
    if (provided < MPI_THREAD_MULTIPLE) {
        printf("MPI version does not support MPI_THREAD_MULTIPLE\n");
        MPI_Finalize();
        return 0;
    }

    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ntasks));

#pragma omp parallel private(msg, tid, nthreads, i)
    {
        nthreads = omp_get_num_threads();
        tid = omp_get_thread_num();

        if (rank == 0) {
#pragma omp single
            {
                printf("%i threads in master rank\n", nthreads);
            }
            for (i = 1; i < ntasks; i++)
                MPI_CHECK(MPI_Send(&tid, 1, MPI_INTEGER, i, tid,
                                   MPI_COMM_WORLD));
        } else {
            MPI_CHECK(MPI_Recv
                      (&msg, 1, MPI_INTEGER, 0, tid, MPI_COMM_WORLD,
                       MPI_STATUS_IGNORE));
            printf("Rank %i thread %i received %i\n", rank, tid, msg);
        }
    }

    MPI_Finalize();
    return 0;
}
