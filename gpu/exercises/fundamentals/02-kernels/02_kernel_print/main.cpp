#include <hip/hip_runtime.h>

__global__ void hello() {
    printf("Hello world, this is thread %d from block %d!\n", threadIdx.x,
           blockIdx.x);
}

int main() {
    const int32_t num_blocks = 1;
    const int32_t num_threads = 1;

    hello<<<num_blocks, num_threads>>>();

    return 0;
}
