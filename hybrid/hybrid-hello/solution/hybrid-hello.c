#include <stdio.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char *argv[])
{
    int my_id, omp_rank;
    int provided, required=MPI_THREAD_FUNNELED;

    MPI_Init_thread(&argc, &argv, required, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

#pragma omp parallel private(omp_rank)
    {
        omp_rank = omp_get_thread_num();
        printf("I'm thread %d in process %d\n", omp_rank, my_id);
    }

    if (my_id == 0) {
        printf("\nProvided thread support level: %d\n", provided);
        printf("  %d - MPI_THREAD_SINGLE\n", MPI_THREAD_SINGLE);
        printf("  %d - MPI_THREAD_FUNNELED\n", MPI_THREAD_FUNNELED);
        printf("  %d - MPI_THREAD_SERIALIZED\n", MPI_THREAD_SERIALIZED);
        printf("  %d - MPI_THREAD_MULTIPLE\n", MPI_THREAD_MULTIPLE);
    }
    MPI_Finalize();
    return 0;
}
