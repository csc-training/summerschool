#include <cstdio>
#include <omp.h>
int main()
{
    printf("Hello world!\n");
#pragma omp parallel
    {
        printf("X\n");
    }
    return 0;
}
