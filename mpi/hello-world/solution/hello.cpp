#include <iostream>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int i, myid, ntasks, namelen;
    char procname[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Get_processor_name(procname, &namelen);

    if (myid == 0) {
        std::cout << "In total there are " << ntasks << " tasks" << std::endl;
    }

    std::cout << "Hello from rank " << myid << " in node " << procname << std::endl;

    MPI_Finalize();
}
