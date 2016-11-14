/* 2D heat equation

   Copyright (C) 2014  CSC - IT Center for Science Ltd.

   Licensed under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   Code is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   Copy of the GNU General Public License can be onbtained from
   see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "heat.h"


int main(int argc, char **argv)
{
    double a = 0.5;             //!< Diffusion constant
    field current, previous;    //!< Current and previous temperature fields

    double dt;                  //!< Time step
    int nsteps = 500;           //!< Number of time steps

    int rows = 200;             //!< Field dimensions with default values
    int cols = 200;

    char input_file[64];        //!< Name of the optional input file

    int image_interval = 10;    //!< Image output interval

    int iter;                   //!< Iteration counter

    double dx2, dy2;            //!< delta x and y squared

    clock_t start_clock;        //!< Time stamps

    /*
     * Following combinations of command line arguments are possible:
     * No arguments:    use default field dimensions and number of time steps
     * One argument:    read initial field from a given file
     * Two arguments:   initial field from file and number of time steps
     * Three arguments: field dimensions (rows,cols) and number of time steps
     */

    switch (argc) {
    case 1:
        /* Use default values */
        initialize_field_metadata(&current, rows, cols);
        initialize_field_metadata(&previous, rows, cols);
        initialize(&current, &previous);
        break;
    case 2:
        /* Read initial field from a file */
        strncpy(input_file, argv[1], 64);
        read_input(&current, &previous, input_file);
        break;
    case 3:
        /* Read initial field from a file */
        strncpy(input_file, argv[1], 64);
        read_input(&current, &previous, input_file);
        /* Number of time steps */
        nsteps = atoi(argv[2]);
        break;
    case 4:
        /* Field dimensions */
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
        initialize_field_metadata(&current, rows, cols);
        initialize_field_metadata(&previous, rows, cols);
        initialize(&current, &previous);
        /* Number of time steps */
        nsteps = atoi(argv[3]);
        break;
    default:
        printf("Unsupported number of command line arguments\n");
        return -1;
    }

    /* Output the initial field */
    output(&current, 0);

    /* Largest stable time step */
    dx2 = current.dx * current.dx;
    dy2 = current.dy * current.dy;
    dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));

    /* Get the start time stamp */
    start_clock = clock();

    /* Time evolve */
    for (iter = 1; iter <= nsteps; iter++) {
        evolve(&current, &previous, a, dt);
        /* Output once in every 10 iterations */
        if (iter % image_interval == 0) {
            output(&current, iter);
        }
        /* Swap current field so that it will be used
         * as previous for next iteration step */
        swap_fields(&current, &previous);
    }

    /* Determine the CPU time used for the iteration */
    printf("Iteration took %.3f seconds.\n", (double)(clock() - start_clock) /
        (double)CLOCKS_PER_SEC);
    printf("Reference value at 5,5: %f\n", previous.data[5][5]);

    finalize(&current, &previous);
    return 0;
}
