#include <stdio.h>
#include <omp.h>

int main(void)
{
    int tid, nthreads;

#pragma omp parallel private(tid) shared(nthreads)
    {
#pragma omp single
        {
            nthreads = omp_get_num_threads();
            printf("There are %i threads in total.\n", nthreads);
        }

        tid = omp_get_thread_num();

#pragma omp critical
        printf("Hello from thread id %i/%i!\n", tid, nthreads);
    }

    return 0;
}
