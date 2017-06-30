#include <stdio.h>
#include <omp.h>

int main(int argc, char argv[]) {
    int omp_rank;
#pragma omp parallel private(omp_rank)
    {
        omp_rank = omp_get_thread_num();
        printf("Hello world! by thread %d\n", omp_rank);
    }
}
