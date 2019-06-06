#include <stdio.h>
#include <mpi.h>

#if !defined(__GNUC__) || defined(_CRAYC)
#error Compile this code with Gnu compiler!
#endif


int main(int argc, char *argv[])
{
    int ntasks, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("Hello world from task %d of %d\n", rank, ntasks);

    MPI_Finalize();
    return 0;
}
