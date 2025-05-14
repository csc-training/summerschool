#include <hip/hip_runtime.h>

__global__ void hello(int32_t num_blocks, int32_t num_threads) {
    printf("Hello world, this is kernel speaking!\n %d * %d = %d threads are "
           "printing this message\n",
           num_blocks, num_threads, num_blocks * num_threads);
}

int main() {
    // TODO: See how the output changes when you change these to different
    // numbers. Once you've experimented with some numbers like 2, 8, 16, try
    // setting num_threads to 1025. What happens?
    const int32_t num_blocks = 1;
    const int32_t num_threads = 1;

    hello<<<num_blocks, num_threads>>>(num_blocks, num_threads);

    return 0;
}
