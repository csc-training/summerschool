/* I/O related functions for heat equation solver */

#include <string>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "heat.hpp"
#include "pngwriter.h"

/* Output routine that prints out a picture of the temperature
 * distribution. */
void write_field(const Field& field, const int iter)
{

    auto height = field.nx;
    auto width = field.ny;

    /* Copy the inner data */
    auto full_data = std::vector<double> (height * width);
    for (int i = 0; i < field.nx; i++) {
        auto start = (i + 1) * (width + 2) + 1;
        auto end = (i + 1) * (width + 2) + 1 + width;
        auto dest = i * width;
        std::copy(field.temperature.begin() + start, field.temperature.begin() + end,
                  full_data.begin() + dest);
    }

    /* Write out the data to a png file */
    std::ostringstream filename_stream;
    filename_stream << "heat_" << std::setw(4) << std::setfill('0') << iter << ".png";
    std::string filename = filename_stream.str();
    save_png(full_data.data(), height, width, filename.c_str(), 'c');
}

/* Read the initial temperature distribution from a file and
 * initialize the temperature fields temperature1 and
 * temperature2 to the same initial state. */
void read_field(Field& field, std::string filename)
{
    std::ifstream file;
    file.open(filename);
    /* Read the header */
    std::string line, comment;
    std::getline(file, line);
    int nx, ny;
    std::stringstream(line) >> comment >> nx >> ny;

    field.setup(nx, ny);

    /* Read the actual data */
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
                auto ind = (i + 1) * (ny+2) + j + 1;
                file >> field.temperature[ind];
        }
    }

    file.close();

    /* Set the boundary values */
    for (int i = 1; i < nx + 1; i++) {
        // left boundary
        auto boundary = i * (ny + 2);
        auto inner = boundary + 1;
        field.temperature[boundary] = field.temperature[inner];
        // right boundary
        boundary = i * (ny + 2) + ny + 1;
        inner = boundary - 1;
        field.temperature[boundary] = field.temperature[inner];
    }
    for (int j = 0; j < ny + 2; j++) {
        // top boundary
        auto boundary = j;
        auto inner = ny + 2 + j;
        field.temperature[boundary] = field.temperature[inner];
        // bottom boundary
        boundary = (nx + 1) * (ny + 2) + j;
        inner = nx * (ny + 2) + j;
        field.temperature[boundary] = field.temperature[inner];
    }

}
