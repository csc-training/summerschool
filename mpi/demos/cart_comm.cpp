#include <cstdio>
#include <mpi.h>

int main(int argc, char *argv[])
{
  int size, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  constexpr int ndims = 2;
  int dims[ndims] = {0};
  int periods[ndims] = {0, 0};

  MPI_Comm comm;
  MPI_Dims_create(size, ndims, dims);
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1, &comm);

  int rank_cart;
  MPI_Comm_rank(comm, &rank_cart);

  // Determine ranks of neighbouring MPI tasks
  int neighbours[ndims][2];
  for (int i=0; i < ndims; i++) {
    MPI_Cart_shift(comm, i, 1, &neighbours[i][0], &neighbours[i][1]);
  }

  printf("Rank: %3d, cart: %3d, left: %3d, right %3d, up %3d, down %3d\n",
         rank, rank_cart,
         neighbours[0][0], neighbours[0][1],
         neighbours[1][0], neighbours[1][1]);

  if (rank == 0) {
    printf("MPI_PROC_NULL: %3d\n", MPI_PROC_NULL);
  }

  MPI_Comm_free(&comm);
  MPI_Finalize();
}

