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
    int tid;

    MPI_CHECK(MPI_Init_thread
              (&argc, &argv, MPI_THREAD_MULTIPLE, &provided));

    /* Check the thread support level */
    if (provided == MPI_THREAD_MULTIPLE) {
        printf("MPI library supports MPI_THREAD_MULTIPLE\n");
    } else if (provided == MPI_THREAD_SERIALIZED) {
        printf("MPI library supports MPI_THREAD_SERIALIZED\n");
    } else if (provided == MPI_THREAD_FUNNELED) {
        printf("MPI library supports MPI_THREAD_FUNNELED\n");
    } else {
        printf("No multithreading support\n");
    }
    

    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ntasks));

#pragma omp parallel private(tid)
    {
        tid = omp_get_thread_num();

        printf("mpi rank: %d  thread id: %d\n", rank, tid);
    }

    MPI_Finalize();
    return 0;
}
