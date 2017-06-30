#include <stdio.h>

int main(void)
{
    int var1 = 1, var2 = 2;

#pragma omp parallel shared(var1, var2)
    {
        printf("Inside region (start): var1=%i, var2=%i\n", var1, var2);
#pragma omp critical(add_first)
        var1++;
#pragma omp atomic
        var2++;
#pragma omp single
        printf("Inside region   (end): var1=%i, var2=%i\n", var1, var2);
    }
    printf("\nAfter region: var1=%i, var2=%i\n\n", var1, var2);

    return 0;
}
