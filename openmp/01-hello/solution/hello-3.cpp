// SPDX-FileCopyrightText: 2010 CSC - IT Center for Science Ltd. <www.csc.fi>
//
// SPDX-License-Identifier: MIT

#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_thread_num()  { return 0; }
static int omp_get_num_threads() { return 1; }
#endif

int main()
{
    printf("Hello world!\n");

    int nthreads;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        printf("Hello from thread %d!\n", tid);

        nthreads = omp_get_num_threads();
    }
    printf("There was %d threads in total!\n", nthreads);

    return 0;
}
