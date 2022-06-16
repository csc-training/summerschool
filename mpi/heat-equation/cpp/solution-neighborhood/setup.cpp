#include <string>
#include <cstdlib>
#include <iostream>
#include "heat.hpp"


void initialize(int argc, char *argv[], Field& current,
                Field& previous, int& nsteps, ParallelData &parallel)
{
    /*
     * Following combinations of command line arguments are possible:
     * No arguments:    use default field dimensions and number of time steps
     * One argument:    read initial field from a given file
     * Two arguments:   initial field from file and number of time steps
     * Three arguments: field dimensions (rows,cols) and number of time steps
     */


    int rows = 2000;             //!< Field dimensions with default values
    int cols = 2000;

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
    case 4:
        /* Field dimensions */
        rows = std::atoi(argv[1]);
        cols = std::atoi(argv[2]);
        /* Number of time steps */
        nsteps = std::atoi(argv[3]);
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
        current.setup(rows, cols, parallel);
        current.generate(parallel);
    }

    // copy "current" field also to "previous"
    previous = current;

}
