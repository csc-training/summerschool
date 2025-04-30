/* I/O related functions for heat equation solver */

#include <string>
#include <iomanip> 
#include <fstream>
#include <sstream>
#include <mpi.h>

#include "heat.hpp"
#include "pngwriter.h"

/* Output routine that prints out a picture of the temperature
 * distribution. */
void write_field(const Field& field, const int iter, const ParallelData parallel)
{

    auto height = field.nx * parallel.size;
    auto width = field.ny;

    // array for MPI sends and receives
    auto tmp_vec = std::vector<double> (field.nx * field.ny); 

    if (parallel.rank == 0) {
        /* Copy the inner data */
        auto full_data = std::vector<double> (height * width);
        for (int i = 0; i < field.nx; i++) {
            auto start = (i + 1) * (width + 2) + 1;
            auto end = (i + 1) * (width + 2) + 1 + width;
            auto dest = i * width;
            std::copy(field.temperature.begin() + start, field.temperature.begin() + end,
                      full_data.begin() + dest);
        }
          
        /* Receive data from other ranks */
        for (int p = 1; p < parallel.size; p++) {
            MPI_Recv(tmp_vec.data(), field.nx * field.ny,
                     MPI_DOUBLE, p, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            /* Copy data to full array */
            std::copy(tmp_vec.begin(), tmp_vec.begin() + field.nx * field.ny,
                      full_data.begin() + p * field.nx * width);
        }
        /* Write out the data to a png file */
        std::ostringstream filename_stream;
        filename_stream << "heat_" << std::setw(4) << std::setfill('0') << iter << ".png";
        std::string filename = filename_stream.str();
        save_png(full_data.data(), height, width, filename.c_str(), 'c');
    } else {
        /* Send data */
        for (int i = 0; i < field.nx; i++) {
            auto start = (i + 1) * (width + 2) + 1;
            auto end = (i + 1) * (width + 2) + 1 + width;
            auto dest = i * width;
            std::copy(field.temperature.begin() + start, field.temperature.begin() + end,
                       tmp_vec.begin() + dest);
        }
        MPI_Send(tmp_vec.data(), field.nx * field.ny,
                 MPI_DOUBLE, 0, 22, MPI_COMM_WORLD);
    }

}

/* Read the initial temperature distribution from a file and
 * initialize the temperature fields temperature1 and
 * temperature2 to the same initial state. */
void read_field(Field& field, std::string filename,
                ParallelData parallel)
{
    std::ifstream file;
    file.open(filename);
    /* Read the header */
    std::string line, comment;
    std::getline(file, line);
    int nx, ny;
    std::stringstream(line) >> comment >> nx >> ny;

    field.setup(nx, ny, parallel);

    /* Full array */
    auto full = std::vector<double> (nx * ny);

    if (parallel.rank == 0) {
        /* Read the actual data */
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
	        auto ind = i * ny + j;
                file >> full[ind];
            }
        }
    }

    file.close();

    auto nx_local = field.nx;
    auto ny_local = field.ny;

    // Inner region (no boundaries)
    auto inner = std::vector<double> (field.nx * field.ny);

    MPI_Scatter(full.data(), nx_local * ny, MPI_DOUBLE, inner.data(),
                nx_local * ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Copy to the array containing also boundaries */
    for (int i = 0; i < nx_local; i++) {
      auto start = i * ny_local;    // beginning of row
      auto end = i * ny_local + ny; // end of row
      auto dest = (i + 1) * (ny_local + 2) + 1; // beginning of inner region
      std::copy(inner.begin() + start, inner.begin() + end,
                field.temperature.begin() + dest);
    }

    /* Set the boundary values */
    for (int i = 1; i < nx_local + 1; i++) {
        // left boundary
        auto boundary = i * (ny_local + 2);
        auto inner = boundary + 1;
        field.temperature[boundary] = field.temperature[inner];
        // right boundary
        boundary = i * (ny_local + 2) + ny + 1;
        inner = boundary - 1;
        field.temperature[boundary] = field.temperature[inner];
    }
    for (int j = 0; j < ny + 2; j++) {
        // top boundary
        auto boundary = j;
        auto inner = ny_local + 2 + j;
        field.temperature[boundary] = field.temperature[inner];
        // bottom boundary
        boundary = (nx_local + 1) * (ny_local + 2) + j;
        inner = nx_local * (ny_local + 2) + j;
        field.temperature[boundary] = field.temperature[inner];
    }

}
