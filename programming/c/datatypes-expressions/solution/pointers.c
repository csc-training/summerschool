#include <stdio.h>

int main(int argc, char *argv[])
{
    int a, b, c;
    int *p;

    a = 4;
    b = 2.1;
    c = 2.0;
    printf("c is %d\n", c);

    p = &c;
    *p = a + 3 * b;
    printf("*p is %d and c is %d\n", *p, c);

    c *= 2;
    printf("*p is %d and c is %d\n", *p, c);

    return 0;
}
