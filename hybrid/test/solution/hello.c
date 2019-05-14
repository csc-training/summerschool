#include <stdio.h>

int main(int argc, char *argv[])
{
    printf("Hello world!\n");
    #pragma omp parallel
    {
        printf("X\n");
    }

    return 0;
}
