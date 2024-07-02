#pragma once
#include "nlohmann/json_fwd.hpp"
#include <string>
#include <tuple>
#include <vector>

namespace heat {
struct Field;
struct ParallelData;

std::tuple<int, int, std::vector<double>>
read_field(const std::string &filename);

void write_field(const Field &field, const ParallelData &parallel,
                 std::string &&filename);

// This corresponds to an input json file
struct Input {
    std::string fname = "";
    std::string png_name_prefix = "heat_";

    double diffusion_constant = 0.5;
    double grid_spacing_x = 0.01;
    double grid_spacing_y = 0.01;

    int rows = 2000;
    int cols = 2000;
    int nsteps = 1000;
    int image_interval = 500;

    bool read_file = false;

    bool operator==(const Input &rhs) const {
        bool equal = true;
        equal &= rows == rhs.rows;
        equal &= cols == rhs.cols;
        equal &= nsteps == rhs.nsteps;
        equal &= image_interval == rhs.image_interval;
        equal &= read_file == rhs.read_file;
        equal &= fname == rhs.fname;

        return equal;
    }

    bool operator!=(const Input &rhs) const { return !(*this == rhs); }
};

void to_json(nlohmann::json &j, const Input &from);
void from_json(const nlohmann::json &j, Input &to);
Input read_input(std::string &&fname, int rank);
std::string make_png_filename(const char *prefix, int iter);
} // namespace heat
