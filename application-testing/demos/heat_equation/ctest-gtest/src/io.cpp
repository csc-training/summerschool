/* I/O related functions for heat equation solver */

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mpi.h>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "field.hpp"
#include "io.hpp"
#include "nlohmann/json.hpp"
#include "parallel.hpp"
#include "pngwriter.h"
#include "utilities.hpp"

namespace heat {
std::tuple<int, int, std::vector<double>>
read_field(const std::string &filename) {
    // Try to open the file
    std::stringstream err_msg;
    std::ifstream file(filename);
    if (!file.is_open()) {
        err_msg << "Could not open file \"" << filename << "\"";
        throw std::runtime_error(err_msg.str());
    }

    // Read the header
    std::string line;
    std::getline(file, line);
    std::string comment;
    int num_rows = 0;
    int num_cols = 0;
    std::stringstream(line) >> comment >> num_rows >> num_cols;

    // Read data to a vector: constructing a vector with istream_iterators reads
    // data from the iterator until the iterator matches the default constructed
    // iterator (i.e. end)
    std::istream_iterator<double> start(file);
    std::istream_iterator<double> end;
    std::vector<double> full_data(start, end);

    // There's an error in the data: the number of elements is not the same as
    // specified in the header
    if (full_data.size() != num_rows * num_cols) {
        err_msg << "size of data(" << full_data.size()
                << ") is not equal to num_rows (" << num_rows
                << ") x num_cols (" << num_cols << "), which is "
                << num_rows * num_cols;
        throw std::runtime_error(err_msg.str());
    }

    return std::make_tuple(num_rows, num_cols, full_data);
}

void write_field(const Field &field, const ParallelData &parallel,
                 std::string &&filename) {
    // Total height and width of the field
    const auto height = field.num_rows * parallel.size;
    const auto width = field.num_cols;
    // Gather data from all MPI processes
    const auto data = gather(field.get_temperatures(), height * width);
    // Only the root saves the data to a png
    if (0 == parallel.rank) {
        save_png(data.data(), height, width, filename.c_str(), 'c');
    }
}

void to_json(nlohmann::json &j, const Input &from) {
    // Convert a structure to a json
    j = nlohmann::json{
        {"fname", from.fname},
        {"png_name_prefix", from.png_name_prefix},
        {"diffusion_constant", from.diffusion_constant},
        {"grid_spacing_x", from.grid_spacing_x},
        {"grid_spacing_y", from.grid_spacing_y},
        {"rows", from.rows},
        {"cols", from.cols},
        {"nsteps", from.nsteps},
        {"image_interval", from.image_interval},
        {"read_file", from.read_file},
    };
}

void from_json(const nlohmann::json &j, Input &to) {
    // Convert a json to a structure
    j.at("fname").get_to(to.fname);
    j.at("png_name_prefix").get_to(to.png_name_prefix);
    j.at("diffusion_constant").get_to(to.diffusion_constant);
    j.at("grid_spacing_x").get_to(to.grid_spacing_x);
    j.at("grid_spacing_y").get_to(to.grid_spacing_y);
    j.at("rows").get_to(to.rows);
    j.at("cols").get_to(to.cols);
    j.at("nsteps").get_to(to.nsteps);
    j.at("image_interval").get_to(to.image_interval);
    j.at("read_file").get_to(to.read_file);
}

Input read_input(std::string &&fname, int rank) {
    // If the given filename is empty, i.e. the user gave no input, use a
    // default constructed Input structure
    std::stringstream ess;
    if (fname.empty()) {
        if (rank == 0) {
            std::cout << "Using default input" << std::endl;
        }
        return Input{};
    }

    // Does the path exist?
    const auto path = std::filesystem::path(fname);
    if (not std::filesystem::exists(path)) {
        ess << "Non-existent path: " << path;
        throw std::runtime_error(ess.str());
    }

    // Can the path be opened?
    std::fstream file(path, std::ios::in);
    if (not file.is_open()) {
        ess << "Could not open file at " << path;
        throw std::runtime_error(ess.str());
    }

    // Read the file to json structure, then convert it to Input
    if (rank == 0) {
        std::cout << "Reading input from " << path << std::endl;
    }
    nlohmann::json j;
    file >> j;
    return j.get<Input>();
}

std::string make_png_filename(const char *prefix, int iter) {
    std::ostringstream filename_stream;
    filename_stream << prefix << std::setw(4) << std::setfill('0') << iter
                    << ".png";
    return filename_stream.str();
}
} // namespace heat
