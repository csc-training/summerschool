// SPDX-FileCopyrightText: 2010 CSC - IT Center for Science Ltd. <www.csc.fi>
//
// SPDX-License-Identifier: MIT

#include <cstdio>
#include <cstdlib>
#include <span>
#include <memory>
#include <omp.h>
#include "helper_functions.hpp"


template <typename T>
auto allocate(size_t n)
{
    T* p = static_cast<T*>(malloc(n * sizeof(T)));
    if (!p) {
        throw std::bad_alloc{};
    }
    return std::unique_ptr<T, void(*)(void*)>(p, &free);
}


void run(const int n)
{
    printf("Array size n = %d\n", n);

    double alpha;

    // Allocate memory (unique pointers, freeing memory automatically)
    auto _x = allocate<double>(n);
    auto _y = allocate<double>(n);

    // Wrap pointer in a span (non-owning, indexable like vector)
    std::span<double> x(_x.get(), n), y(_y.get(), n);

    // Initialization
    alpha = 3.0;
    #pragma omp parallel for
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

    // Start timing
    double t0 = omp_get_wtime();

    // Calculate axpy
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        y[i] += alpha * x[i];
    }

    // End timing
    double t1 = omp_get_wtime();

    // Print output values
    printf("Output:\n");
    print_array("y", y);
    printf("Calculating axpy took %.3f milliseconds\n", (t1 - t0) * 1e3);
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
