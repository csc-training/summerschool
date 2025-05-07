#include <charconv>
#include <iostream>
#include <map>
#include <string_view>
#include <vector>

#include "axpy.h"
#include "common.h"
#include "taylor.h"

template <typename T> void get_arg(const char *arg, T &value) {
    auto [ptr, ec] = std::from_chars(arg, arg + sizeof(arg), value);
    assert(ec == std::errc() && "Failed to read argument");
};

std::vector<std::string_view> split(std::string_view str,
                                    std::string_view sep) {
    std::vector<std::string_view> out;
    while (1) {
        const auto pos = str.find(sep);
        out.push_back(str.substr(0, pos));
        if (pos == str.npos) {
            break;
        }
        str = str.substr(pos + 1);
    }

    return out;
}

void handle_input(int argc, char **argv) {
    const std::map<std::string_view, std::vector<std::string_view>> subprogs = {
        {"axpy", {"a"}},
        {"taylor", {"x_min", "x_max", "num_iters"}},
    };

    if (argc <= 1) {
        std::cerr << "Use one of the following commands:";
        for (const auto &[key, values] : subprogs) {
            std::cerr << "\n\t" << argv[0] << " " << key;
            for (const auto &value : values) {
                std::cerr << ":" << value;
            }
        }
        std::cerr << std::endl;
    } else {
        // TODO figure out this cluster fck
        const std::vector<std::string_view> args = split(argv[1], ":");
        // const std::vector<std::string_view> arguments = subprogs[args[0]];

        auto n = 0;

        for (const auto &[subprog, subprog_args] : subprogs) {
            if (0 == subprog.compare(args[0])) {
                if (subprog_args.size() != args.size() - 1) {
                    std::cerr << "Incorrect number of arguments for subprogram "
                              << subprog << std::endl;
                    std::cerr << "Given arguments: " << argv[1];
                    return;
                }

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
    // handle_input(argc, argv);
    run_and_measure<Taylor<float>>(1.0, 10.0, 15);
    // run_and_measure<Axpy<float>>(1.2);

    return 0;
}
