#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {

    // Global MPI initialization, must be paired with MPI_Finalize at end of the program
    MPI_Init(&argc, &argv);

    // Query size of MPI "world", ie. all copies of the program that were started by mpirun/srun
    int ntasks;
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    // Find the identifier (rank) of this process within the MPI world
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        printf("In total there are %i tasks\n", ntasks);
    }

    // Bonus: find name of the processor (node) that this rank is running on.
    // Docs: https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man3/MPI_Get_processor_name.3.html.
    // As stated in the docs, we must allocate a char array of at least length MPI_MAX_PROCESSOR_NAME.

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    // Will hold the actual length of the processor name (filled in by MPI)
    int processor_name_length;

    MPI_Get_processor_name(processor_name, &processor_name_length);

    printf("Hello from rank %i in node %s\n", rank, processor_name);

    MPI_Finalize();
}
