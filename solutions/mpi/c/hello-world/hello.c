#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>

int main(int argc, char *argv[])
{
    int i, myid, ntasks;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (myid == 0) {
        printf("In total there are %i tasks\n", ntasks);
    }

    printf("Hello from %i\n", myid);

    MPI_Finalize();
    return 0;
}
