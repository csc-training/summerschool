// SPDX-FileCopyrightText: 2009 CSC - IT Center for Science Ltd. <www.csc.fi>
//
// SPDX-License-Identifier: MIT

#include <cstdio>

int main()
{
    printf("Hello world!\n");

    #pragma omp parallel
    {
        printf("Hello from thread!\n");
    }

    return 0;
}
