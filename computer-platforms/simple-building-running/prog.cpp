#include <stdio.h>
#include <mpi.h>

#if !defined(__GNUC__) || defined(_CRAYC)
#error Compile this code with Gnu compiler!
#endif

void print_hello();

int main(int argc, char *argv[])
{
    int ntasks, rank;

    MPI_Init(&argc, &argv);

    print_hello();

    MPI_Finalize();
    return 0;
}
