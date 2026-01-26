// SPDX-FileCopyrightText: 2010 CSC - IT Center for Science Ltd. <www.csc.fi>
//
// SPDX-License-Identifier: MIT

#include <cstdio>
#include <vector>
#include "helper_functions.hpp"


void run(const int n)
{
    printf("Array size n = %d\n", n);

    double alpha;
    std::vector<double> x(n), y(n);

    // Initialization
    alpha = 3.0;
    for (int i = 0; i < n; i++) {
        double frac = 1.0 / ((double) (n - 1));
        x[i] = i * frac;
        y[i] = i * frac * 100;
    }

    // Print input values
    printf("Input:\n");
    printf("a = %8.4f\n", alpha);
    print_array("x", x);
    print_array("y", y);

    // Calculate axpy
    // TODO: Add OpenMP directives for parallelization
    for (int i = 0; i < n; i++) {
        y[i] += alpha * x[i];
    }

    // Print output values
    printf("Output:\n");
    print_array("y", y);
}

int main(int argc, char *argv[])
{
    // Array size
    int n = 102400;

    if (argc > 1) {
        n = atoi(argv[1]);
        if (n < 1) {
            printf("Size needs to be greater than zero.\n");
            return 1;
        }
    }

    run(n);

    return 0;
}
