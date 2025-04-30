#include <iostream>
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
        std::cout << "In total there are " << ntasks << " tasks" << std::endl;
    }

    // Bonus: find name of the processor (node) that this rank is running on.
    // As stated in the docs for MPI_Get_processor_name,
    // we must allocate a char array of at least length MPI_MAX_PROCESSOR_NAME.

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    // Will hold the actual length of the processor name (filled in by MPI)
    int processor_name_length;

    MPI_Get_processor_name(processor_name, &processor_name_length);

    std::cout << "Hello from rank " << rank << " in node " << processor_name << std::endl;

    MPI_Finalize();
}
