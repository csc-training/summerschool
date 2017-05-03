#include "functions.h"

// Main program: initialise, compute Laplacian and print temperature field.
int main(void)
{
    field f;

    init_field(&f, NX, NY);
    laplacian(&f);
    print_field(&f);

    return 0;
}
