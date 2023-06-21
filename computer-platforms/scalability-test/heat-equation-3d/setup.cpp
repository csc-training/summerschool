#include <string>
#include <cstdlib>
#include <iostream>
#include "parallel.hpp"
#include "heat.hpp"
#include "functions.hpp"


void initialize(int argc, char *argv[], Field& current,
                Field& previous, int& nsteps, ParallelData& parallel)
{
    /*
     * Following combinations of command line arguments are possible:
     * No arguments:    use default field dimensions and number of time steps
     * One argument:    read initial field from a given file
     * Two arguments:   initial field from file and number of time steps
     * Three arguments: field dimensions (rows,cols) and number of time steps
     */


    int height = 800;             //!< Field dimensions with default values
    int width = 800;
    int length = 800;

    std::string input_file;        //!< Name of the optional input file

    bool read_file = 0;

    nsteps = 500;

    switch (argc) {
    case 1:
        /* Use default values */
        break;
    case 2:
        /* Read initial field from a file */
        input_file = argv[1];
        read_file = true;
        break;
    case 3:
        /* Read initial field from a file */
        input_file = argv[1];
        read_file = true;

        /* Number of time steps */
        nsteps = std::atoi(argv[2]);
        break;
    case 5:
        /* Field dimensions */
        height = std::atoi(argv[1]);
        width = std::atoi(argv[2]);
        length = std::atoi(argv[3]);
        /* Number of time steps */
        nsteps = std::atoi(argv[4]);
        break;
    default:
        std::cout << "Unsupported number of command line arguments" << std::endl;
        exit(-1);
    }

    if (read_file) {
        if (0 == parallel.rank)
            std::cout << "Reading input from " + input_file << std::endl;
        read_field(current, input_file, parallel);
    } else {
        current.setup(height, width, length, parallel);
        current.generate(parallel);
    }

    // copy "current" field also to "previous"
    previous = current;

    if (0 == parallel.rank) {
        std::cout << "Simulation parameters: " 
                  << "height: " << height << " width: " << width << " length: " << length
                  << " time steps: " << nsteps << std::endl;
        std::cout << "Number of MPI tasks: " << parallel.size 
                  << " (" << parallel.dims[0] << " x " << parallel.dims[1] << " x " 
                  << parallel.dims[2] << ")" << std::endl;
        std::cout << "Number of GPUs per node: " << parallel.dev_count << std::endl;
       #ifndef NO_MPI
        #if defined MPI_DATATYPES && MPI_NEIGHBORHOOD
        std::cout << "Both MPI_DATATYPES and MPI_NEIGHBORHOOD defined; "
                  << "using isend/irecv with datatypes in communication" << std::endl;
        #elif defined MPI_DATATYPES
        std::cout << "Using isend/irecv with datatypes in communication" << std::endl;
        #elif defined MPI_NEIGHBORHOOD
        std::cout << "Using neighborhood collective in communication" << std::endl;
        #else
        std::cout << "Using manual packing of send/recv buffers" << std::endl;
        #endif
        #endif
    }
}
