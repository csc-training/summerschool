#include <hip/hip_runtime.h>

__global__ void hello(int32_t num_blocks, int32_t num_threads) {
    printf("Hello world, this is kernel speaking!\n %d * %d = %d threads are "
           "printing this message\n",
           num_blocks, num_threads, num_blocks * num_threads);
}

int main() {
    // TODO 1:
    // - compile and run the code

    // TODO 2:
    // - change num_blocks and/or num_threads to different numbers
    // - use small numbers to make some sense of the output

    // TODO 3:
    // - change num_threads to 1025. What happens?
    // - Move on to the next exercise to find out more.

    const int32_t num_blocks = 1;
    const int32_t num_threads = 1;

    hello<<<num_blocks, num_threads>>>(num_blocks, num_threads);

    return 0;
}
