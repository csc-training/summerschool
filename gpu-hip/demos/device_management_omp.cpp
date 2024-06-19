#include <cstdio>
#include <omp.h>

int main(int argc, char *argv[])
{
    int count, device;

    count = omp_get_num_devices();
    device = omp_get_default_device();

    printf("Hello! I'm GPU %d out of %d GPUs in total.\n", device, count);

    omp_set_default_device(count - 1);
    device = omp_get_default_device();
    printf("Now I'm GPU %d.\n", device);

    return 0;
}
