/*
 * This example is based on the code of Andrew V. Adinetz
 * https://github.com/canonizer/mandelbrot-dyn
 * Licensed under The MIT License
 */

#include <omp.h>
#include <complex>
#include <cstdio>
#include <cstdlib>

#include "pngwriter.h"

using namespace std;

// Maximum number of iterations
const int MAX_ITER_COUNT = 512;
// Marker for different iteration counts
const int DIFF_ITER_COUNT = -1;
// Maximum recursion depth
const int MAX_DEPTH = 6;
// Region size below which do per-pixel
const int MIN_SIZE = 32;
// Subdivision factor along each axis
const int SUBDIV = 4;


// |z|^2 of a complex number z
float abs2(complex<double> v)
{
    return v.real() * v.real() + v.imag() * v.imag();
}

// The kernel to count per-pixel values of the portion of the Mandelbrot set
// Does not need to be edited
int kernel(int w, int h, complex<double> cmin, complex<double> cmax,
           int x, int y)
{
    complex<double> dc = cmax - cmin;
    float fx = (float)x / w;
    float fy = (float)y / h;
    double real = cmin.real() + fx * dc.real();
    double imag = cmin.imag() + fy *dc.imag();
    complex<double> c(real, imag);
    int iteration = 0;
    complex<double> z = c;
    while (iteration < MAX_ITER_COUNT && abs2(z) < 2 * 2) {
        z = z * z + c;
        iteration++;
    }
    return iteration;
}


/* Computes the Mandelbrot image recursively
 * At each call, the image is divided into smaller blocks (by a factor of
 * subdiv), and the function is called recursively with arguments corresponding
 * to subblock. When maximum recursion depth is reached or size of block
 * is smaller than predefined minimum, one starts to calculate actual pixel
 * values
 *
 * - - - - - - - -           -----       -----
 * |             |           |   |       |   |
 * |             |           -----       -----
 * |             |  -->                         -->   ...
 * |             |           -----       -----
 * |             |           |   |       |   |
 * |             |           -----       -----
 * ---------------
 */
void mandelbrot_block(int *iter_counts, int w, int h, complex<double> cmin,
                      complex<double> cmax, int x0, int y0, int d, int depth)
{

// TODO Parallelize the recursive function call
// with OpenMP tasks

    int block_size = d / SUBDIV;
    if (depth + 1 < MAX_DEPTH && block_size > MIN_SIZE) {
        // Subdivide recursively
        for (int i = 0; i < SUBDIV; i++) {
            for (int j = 0; j < SUBDIV; j++) {
                #pragma omp task
                mandelbrot_block(iter_counts, w, h, cmin, cmax,
                                 x0 + i * block_size, y0 + j * block_size,
                                 d / SUBDIV, depth + 1);
            }
        }
    } else {
        // Last recursion level reached, calculate the values
        for (int i = x0; i < x0 + d; i++) {
            for (int j = y0; j < y0 + d; j++) {
                iter_counts[j * w + i] = kernel(w, h, cmin, cmax, i, j);
            }
        }
    }
}


int main(int argc, char **argv)
{
    // Picture size, should be power of two
    const int w = 512;
    const int h = w;
    int *iter_counts;

    int pic_bytes = w * h * sizeof(int);
    iter_counts = (int *)malloc(pic_bytes);

    complex<double> cmin(-1.5, -1.0);
    complex<double> cmax(0.5, 1.0);

    double t1 = omp_get_wtime();

// TODO create parallel region. How many threads should be calling
// mandelbrot_block in this uppermost level?
    #pragma omp parallel
    #pragma omp single
    {
        mandelbrot_block(iter_counts, w, h, cmin, cmax,
                         0, 0, w, 1);
    }
    double t2 = omp_get_wtime();

    // Save the image to a PNG file
    save_png(iter_counts, w, h, "mandelbrot.png");

    double walltime = t2 - t1;
    // Print the timings
    printf("Mandelbrot set computed in %.3lf s, at %.3lf Mpix/s\n",
           walltime, h * w * 1e-6 / walltime);

    free(iter_counts);
    return 0;
}
