#include <stdio.h>

int main(int argc, char *argv[])
{
    int f[20];
    int i;

    f[0] = 0;
    f[1] = 1;
    for (i = 2; i < 20; i++) {
        f[i] = f[i - 2] + f[i - 1];
    }

    printf("First 20 Fibonacci numbers are:\n");
    for (i = 0; i < 20; i++) {
        printf("%d ", f[i]);
    }

    printf("\n");

    return 0;
}
