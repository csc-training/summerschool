#include <stdio.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    int rank, thread_id;
    int provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#pragma omp parallel private(thread_id)
{
    thread_id = omp_get_thread_num();
    printf("I'm thread %d in process %d\n", thread_id, rank);
}

    MPI_Finalize();
}

