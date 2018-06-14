#include <stdio.h>

int main(int argc, char *argv[])
{
    int a, b, c;
    float d, e, f;
    double g, h, o;

    a = 4;
    b = 2.1;
    c = a + 3 * b;
    printf("c is %d\n", c);

    d = 4.1;
    e = -3;
    f = d - 2 / e;
    printf("f is %.9f\n", f);

    g = 4.1;
    h = -3;
    o = g - 2.0 / e;
    printf("o is %.9f\n", o);

    a = 2 * d - f / 5.6;
    printf("a is %d\n", a);

    return 0;
}
