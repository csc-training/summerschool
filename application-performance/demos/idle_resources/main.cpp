#include "common.h"
#include <array>
#include <charconv>
#include <iostream>
#include <string_view>

#include "axpy.h"
#include "taylor.h"

void handle_input(int argc, char **argv) {
    static constexpr std::array<std::string_view, 2> arguments = {
        "axpy a",
        "taylor range_min range_max num_iters",
    };

    if (argc <= 1) {
        std::cerr << "Usage:";
        for (size_t i = 0; i < arguments.size() - 1; i++) {
            std::cerr << "\n\t" << argv[0] << " " << arguments[i] << " or";
        }
        std::cerr << "\n\t" << argv[0] << " " << arguments[arguments.size() - 1]
                  << std::endl;
    } else {
        std::string input(argv[1]);
        auto n = 0;
        for (auto &arg : arguments) {
            auto substr = arg.substr(0, arg.find(' '));
            if (0 == input.compare(substr)) {
                auto get_arg = [](const char *arg, auto &value) {
                    auto [ptr, ec] =
                        std::from_chars(arg, arg + sizeof(arg), value);
                    assert(ec == std::errc() && "Failed to read argument");
                };

                switch (n) {
                case 0: {
                    assert(argc == 3 && "Incorrect amount of arguments");
                    float a = 0.0f;
                    get_arg(argv[2], a);
                    std::cerr << "Running axpy with argument " << a
                              << std::endl;
                    run_and_measure<Axpy<float>>(a);
                    break;
                }
                case 1: {
                    assert(argc == 5 && "Incorrect amount of arguments");
                    float min = 0;
                    float max = 0;
                    size_t num_iters = 0;

                    get_arg(argv[2], min);
                    get_arg(argv[3], max);
                    get_arg(argv[4], num_iters);

                    std::cerr << "Running taylor with arguments " << min << ", "
                              << max << ", " << num_iters << std::endl;
                    run_and_measure<Taylor<float>>(min, max, num_iters);
                    break;
                }
                default:
                    std::cerr << "Index out of range" << std::endl;
                    break;
                }
                return;
            }
            n++;
        }
    }

    std::cerr << "Given subprogram name \"" << argv[1] << "\" didn't match"
              << std::endl;
}

int main(int argc, char **argv) {
    handle_input(argc, argv);

    return 0;
}
