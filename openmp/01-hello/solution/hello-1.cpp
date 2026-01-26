// SPDX-FileCopyrightText: 2010 CSC - IT Center for Science Ltd. <www.csc.fi>
//
// SPDX-License-Identifier: MIT

#include <cstdio>
#include <omp.h>

int main()
{
    printf("Hello world!\n");

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        printf("Hello from thread %d!\n", tid);
    }

    return 0;
}
