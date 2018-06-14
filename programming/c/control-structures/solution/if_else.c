#include <stdio.h>

int main(int argc, char *argv[])
{
    int i;

    i = 12;

    if (i < 0) {
        printf("i is negative %d\n", i);
    } else if (i == 0) {
        printf("i is zero %d\n", i);
    } else if (i > 100) {
        printf("i is large %d\n", i);
    } else {
        printf("i is something else %d\n", i);
    }

    return 0;
}
