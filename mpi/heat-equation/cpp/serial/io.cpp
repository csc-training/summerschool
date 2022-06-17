/* I/O related functions for heat equation solver */

#include <string>
#include <iomanip> 
#include <fstream>
#include <string>

#include "matrix.hpp"
#include "heat.hpp"
#include "pngwriter.h"

// Write a picture of the temperature field
void write_field(const Field& field, const int iter)
{

    auto height = field.nx * 1;
    auto width = field.ny;

    // array for MPI sends and receives
    auto tmp_mat = Matrix<double> (field.nx, field.ny); 

    // Copy the inner data
    auto full_data = Matrix<double>(height, width);
    for (int i = 0; i < field.nx; i++)
        for (int j = 0; j < field.ny; j++) 
             full_data(i, j) = field(i + 1, j + 1);
          
    // Write out the data to a png file 
    std::ostringstream filename_stream;
    filename_stream << "heat_" << std::setw(4) << std::setfill('0') << iter << ".png";
    std::string filename = filename_stream.str();
    save_png(full_data.data(), height, width, filename.c_str(), 'c');

}

// Read the initial temperature distribution from a file
void read_field(Field& field, std::string filename)
{
    std::ifstream file;
    file.open(filename);
    // Read the header
    std::string line, comment;
    std::getline(file, line);
    int nx, ny;
    std::stringstream(line) >> comment >> nx >> ny;

    field.setup(nx, ny);

    // Inner region (no boundaries)
    auto inner = Matrix<double> (field.nx, field.ny);

    // Read the array
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            file >> inner(i, j);

    file.close();

    // Copy to the array containing also boundaries
    for (int i = 0; i < field.nx; i++)
        for (int j = 0; j < field.ny; j++)
             field(i + 1, j + 1) = inner(i, j);

    // Set the boundary values
    for (int i = 0; i < field.nx + 2; i++) {
        // left boundary
        field(i, 0) = field(i, 1);
        // right boundary
        field(i, field.ny + 1) = field(i, field.ny);
    }
    for (int j = 0; j < field.ny + 2; j++) {
        // top boundary
        field.temperature(0, j) = field(1, j);
        // bottom boundary
        field.temperature(field.nx + 1, j) = field.temperature(field.nx, j);
    }

}
