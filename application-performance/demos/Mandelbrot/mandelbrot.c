/* 
 * This example is based on the code of Andrew V. Adinetz
 * https://github.com/canonizer/mandelbrot-dyn
 * Licensed under The MIT License
 */

#include <mpi.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>

#include "pngwriter.h"

// Maximum number of iterations
const int MAX_ITER_COUNT=512;
// Marker for different iteration counts
const int DIFF_ITER_COUNT = -1;
// Maximum recursion depth
const int MAX_DEPTH = 6;
// Region size below which do per-pixel
const int MIN_SIZE = 32;
// Subdivision factor along each axis
const int SUBDIV = 4;


// |z|^2 of a complex number z
float abs2(complex v)
{
    return creal(v) * creal(v) + cimag(v) * cimag(v);
}

// The kernel to count per-pixel values of the portion of the Mandelbrot set
// Does not need to be edited
int kernel(int w, int h, complex cmin, complex cmax,
        int x, int y)
{
    complex dc = cmax - cmin;
    float fx = (float)x / w;
    float fy = (float)y / h;
    complex c = cmin + fx * creal(dc) + fy * cimag(dc) * I;
    int iteration = 0;
    complex z = c;
    while(iteration < MAX_ITER_COUNT && abs2(z) < 2 * 2) {
        z = z * z + c;
        iteration++;
    }
    return iteration;
} 


int main(int argc, char **argv)
{
    // Picture size, should be power of two
    const int w = 256;
    const int h = w;
    int *iter_counts, *my_iter_counts;

    int ntasks, myid;
    int myh;

    complex cmin, cmax;

    int pic_bytes = w * h * sizeof(int);
    iter_counts = (int*)malloc(pic_bytes);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    cmin = -1.5 + -1.0*I;
    cmax = 0.5 + 1.0*I;

    /* Very simple parallelisation, where the domain is divided evenly 
     * along one dimension. Because computation cost varies within the domain
     * the code demonstrates how load imbalance hinders parallel scalability
    */
    myh = h / ntasks;
    my_iter_counts = malloc(w * myh *  sizeof(int));

    double t1 = MPI_Wtime();

    for (int i = 0; i < w; i++) {
      for (int j = 0; j < myh; j++) {
	my_iter_counts[j * w + i] = kernel(w, h, cmin, cmax, i, j + myid*myh);
      }
    }

    MPI_Gather(my_iter_counts, w*myh, MPI_INT, iter_counts, w*myh, MPI_INT, 0,
	       MPI_COMM_WORLD);

    double t2 = MPI_Wtime();

    if (myid == 0)
      {
	// Save the image to a PNG file
	save_png(iter_counts, w, h, "mandelbrot.png");

	double walltime = t2 - t1;
	// Print the timings
	printf("Mandelbrot set computed in %.3lf s, at %.3lf Mpix/s\n",
	       walltime, h * w * 1e-6 / walltime );
      }


    free(iter_counts);
    free(my_iter_counts);
    
    MPI_Finalize();

    return 0;
}

