#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[])
{
    int tid, nthreads;

    printf("Hello world!\n");
    #pragma omp parallel private(tid) shared(nthreads)
    {
        tid = omp_get_thread_num();
        #pragma omp single
        nthreads = omp_get_num_threads();

        #pragma omp critical
        printf("  ... from thread ID %i.\n", tid);
    }
    printf("There were %i threads in total.\n", nthreads);

    return 0;
}
