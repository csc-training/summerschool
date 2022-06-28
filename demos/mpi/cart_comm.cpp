#include <cstdio>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int myid, ntasks;

    int source, destination;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    MPI_Comm cart_comm;
    int ndims = 1;
    int dims[1] = {ntasks};
    int periods[1] = {1};

    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1, &cart_comm);
    MPI_Cart_shift(cart_comm, 0, 1, &source, &destination);

    printf("My id: %d left: %d right %d\n", myid, source, destination);

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}

