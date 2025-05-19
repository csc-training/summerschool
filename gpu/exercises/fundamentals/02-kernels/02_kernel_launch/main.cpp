#include <hip/hip_runtime.h>

__global__ void hello(int32_t num_blocks, int32_t num_threads) {
    assert(num_blocks != 2);

    printf("Hello world, this is kernel speaking!\n %d * %d = %d threads are "
           "printing this message\n",
           num_blocks, num_threads, num_blocks * num_threads);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Give number of blocks and number of threads as arguments\n");
        printf("For example \"%s 1 8\"\n", argv[0]);
    }
    const int32_t num_blocks = std::atoi(argv[1]);
    const int32_t num_threads = std::atoi(argv[2]);

    printf("Launching with %d blocks and %d threads\n", num_blocks,
           num_threads);
    hello<<<num_blocks, num_threads>>>(num_blocks, num_threads);

    return 0;
}
