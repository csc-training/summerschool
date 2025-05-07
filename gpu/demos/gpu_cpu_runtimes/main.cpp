#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "common.h"
#include "taylor.h"

size_t get_num_iters(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(
            stderr,
            "Give number of Taylor series iterations as input argument\n");
        std::fprintf(stderr, "E.g.: %s 10\n", argv[0]);

        return EXIT_FAILURE;
    }

    size_t num_iters = 0;
    const auto [ptr, ec] =
        std::from_chars(argv[1], argv[1] + std::strlen(argv[1]), num_iters);
    if (ec != std::errc()) {
        std::fprintf(stderr,
                     "Failed to convert first input argument to size_t. Given "
                     "argument: %s\n",
                     argv[1]);

        return EXIT_FAILURE;
    }

    return num_iters;
}

void check_for_correctness(size_t num_iters) {
    Taylor<float> taylor(10, std::malloc, std::free, 1.0f, 10.0f, num_iters);
    for (size_t i = 0; i < taylor.size; i++) {
        taylor.init(i);
        taylor.compute(i);
        std::printf("%f, %f, %f\n", taylor.x[i], taylor.y[i],
                    taylor.y[i] - exp(taylor.x[i]));
    }
}

int main(int argc, char **argv) {
    const size_t num_iters = get_num_iters(argc, argv);
    run_and_measure<Taylor<float>>(1.0f, 10.0f, num_iters);
    // check_for_correctness(num_iters);

    return 0;
}
