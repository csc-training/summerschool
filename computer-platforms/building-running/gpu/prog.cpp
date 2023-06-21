#include <stdio.h>
#include <mpi.h>

void print_hello();

int main(int argc, char *argv[])
{
    int ntasks, rank;

    MPI_Init(&argc, &argv);

    print_hello();

    MPI_Finalize();
    return 0;
}
