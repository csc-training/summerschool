#include <stdio.h>

int main(int argc, char *argv[])
{
    int f0, f1, f2;

    f0 = 0;
    f1 = 1;
    printf("%d\n%d\n", f0, f1);
    f2 = f0 + f1;
    while (f2 < 100) {
        printf("%d\n", f2);
        f0 = f1;
        f1 = f2;
        f2 = f0 + f1;
    }

    return 0;
}
