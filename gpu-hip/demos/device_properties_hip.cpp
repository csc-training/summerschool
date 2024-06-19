#include <cstdio>
#include <cmath>
#include <hip/hip_runtime.h>

int main(int argc, char *argv[])
{
    int count, device;

    hipGetDeviceCount(&count);
    hipGetDevice(&device);

    printf("Hello! I'm GPU %d out of %d GPUs in total.\n", device, count);

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device);

    // Note: name is empty string on LUMI, see https://github.com/ROCm/ROCm/issues/1625
    printf("Name: %s\n", prop.name);
    printf("Memory: %.2f GiB\n", prop.totalGlobalMem / pow(1024., 3));
    printf("Wavefront / warp size: %d\n", prop.warpSize);

    return 0;
}
