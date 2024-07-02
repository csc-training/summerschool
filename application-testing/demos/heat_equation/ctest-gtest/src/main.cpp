/* Heat equation solver in 2D. */

#include <mpi.h>
#include <string>

namespace heat {
void run(std::string &&);
}

int main(int argc, char **argv)
{
    // Initialize MPI
    // Get first argument (if any) from cmdline
    // Call the library
    // Finalize MPI
    // Return
    MPI_Init(&argc, &argv);
    std::string fname = "";
    if (argc > 1) {
        fname = argv[1];
    }
    heat::run(std::move(fname));
    MPI_Finalize();

    return 0;
}
