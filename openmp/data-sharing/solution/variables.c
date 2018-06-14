#include <stdio.h>

int main(void)
{
    int var1 = 1, var2 = 2;

    #pragma omp parallel private(var1, var2)
    {
        printf("Region 1: var1=%i, var2=%i\n", var1, var2);
        var1++;
        var2++;
    }
    printf("After region 1: var1=%i, var2=%i\n\n", var1, var2);

    #pragma omp parallel firstprivate(var1, var2)
    {
        printf("Region 2: var1=%i, var2=%i\n", var1, var2);
        var1++;
        var2++;
    }
    printf("After region 2: var1=%i, var2=%i\n\n", var1, var2);

    #pragma omp parallel            /* same as omp parallel shared(var1, var2) */
    {
        printf("Region 3: var1=%i, var2=%i\n", var1, var2);
        /* Note that this introduces the data race condition! */
        var1++;
        var2++;
    }
    printf("After region 3: var1=%i, var2=%i\n\n", var1, var2);

    return 0;
}
