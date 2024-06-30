#include <cstdio>
#include <omp.h>
int main()
{
    int num_threads;
    printf("Hello world!\n");
#pragma omp parallel
    {
        num_threads = omp_get_num_threads();
        printf("Thread number: %d\n", omp_get_thread_num());
    }
    printf("Total number of threads: %d\n", num_threads);
    return 0;
}


// Compile with `cc -fopenmp hello.cpp -o omp_hello.exe` to enable openMP.