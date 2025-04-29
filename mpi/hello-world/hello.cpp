#include <iostream>
#include <mpi.h>

int main(int argc, char *argv[]) {

    // TODO: say hello! in parallel

    MPI_Init(&argc, &argv);

    int myrank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (myrank == 0) {
        std::cout << "World size is " << world_size << "\n";
    }

    // Bonus: get node name
    char cpu_name[MPI_MAX_PROCESSOR_NAME];
    int cpu_name_len;
    MPI_Get_processor_name(cpu_name, &cpu_name_len);

    std::cout << "Hello from rank " <<  myrank << std::endl;
    std::cout << "Rank " << myrank << " has CPU: " << cpu_name << "\n";

    MPI_Finalize();
}
