#include <charconv>
#include <cstdio>
#include <cstring>

#include "common.h"
#include "taylor.h"

int main(int argc, char **argv) {
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

    run_and_measure<Taylor<float>>(1.0f, 10.0f, num_iters);

    return 0;
}
