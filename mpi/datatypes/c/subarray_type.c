#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank;
    int array[8][8];
    //TODO: declare mpi datatype for the block
    int sizes[2], subsizes[2], offsets[2];

    int i, j;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialize arrays
    if (rank == 0) {
        for (i = 0; i < 8; i++) {
            for (j = 0; j < 8; j++) {
                array[i][j] = (i + 1) * 10 + j + 1;
            }
        }
    } else {
        for (i = 0; i < 8; i++) {
            for (j = 0; j < 8; j++) {
                array[i][j] = 0;
            }
        }
    }

    //TODO: Create datatype for a subblock [2:5][3:5] of the 8x8 matrix
/*
    sizes[0] = 
    sizes[1] = 
    subsizes[0] = 
    subsizes[1] = 
    offsets[0] = 
    offsets[1] = 
*/
    

    // TODO:  Send a block of a matrix using the user-defined datatype from
    // rank 0 to rank 1

    // Print out the result on rank 1
    if (rank == 1) {
        for (i = 0; i < 8; i++) {
            for (j = 0; j < 8; j++) {
                printf("%3d", array[i][j]);
            }
            printf("\n");
        }
    }

    //TODO: free datatype
    MPI_Finalize();

    return 0;
}
