/*
 * SPDX-FileCopyrightText: 2025 CSC - IT Center for Science Ltd. <www.csc.fi>
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstdio>
#include <cmath>

#ifdef TRACE
#include <roctx.h>
#else
#define roctxRangePush(...) ((void)0)
#define roctxRangePop(...) ((void)0)
#endif


template <typename T>
static void print_array(const char* name, const T& x)
{
    const size_t PRINT_N = 4; // Limit the print size
    const size_t n = size(x);
    printf("%s =", name);
    for (size_t i = 0; i < n; i++) {
        if (i < PRINT_N || i > n - 1 - PRINT_N) {
            printf(" %8.4f", x[i]);
        } else if (i == PRINT_N) {
            printf(" ...");
        }
    }
    printf("\n");
}


static
void create_input(double *f, int nx, int ny) {
    double cx = nx / 2.0;
    double cy = ny / 2.0;
    double sigma2 = 0.05 * nx*ny;  // Width of the Gaussian
    double kx = 20.0 / nx;  // Spatial frequency in x
    double ky = 10.0 / ny;  // Spatial frequency in y

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            int ind = i * nx + j;

            double dx = j - cx;
            double dy = i - cy;
            double r2 = dx * dx + dy * dy;

            f[ind] = cos(kx * dx + ky * dy) * exp(-r2 / sigma2);
        }
    }
}


static
int write_array(const char *filename, const double *array, size_t count)
{
    roctxRangePush(__func__);

    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        perror("Failed to open file");
        roctxRangePop();
        return 1;
    }

    // Write the array size
    fwrite(&count, sizeof(size_t), 1, file);

    // Write the array data
    size_t written = fwrite(array, sizeof(double), count, file);

    fclose(file);

    if (written != count) {
        fprintf(stderr, "Failed to write all elements to file\n");
        roctxRangePop();
        return 2;
    }

    roctxRangePop();

    return 0;
}

