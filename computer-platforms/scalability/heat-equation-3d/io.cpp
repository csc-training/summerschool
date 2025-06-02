/* I/O related functions for heat equation solver */

#include <string>
#include <iomanip> 
#include <fstream>
#include <iostream>
#include <string>
#ifndef NO_MPI
#include <mpi.h>
#endif
#include "matrix.hpp"
#include "heat.hpp"
#include "parallel.hpp"
#ifndef DISABLE_PNG
#include "pngwriter.h"
#endif

// Write a picture of the temperature field
void write_field(Field& field, const int iter, const ParallelData& parallel)
{

    auto height = field.nx_full;
    auto width = field.ny_full;
    auto length = field.nz_full;

    if (0 == parallel.rank) {
        // Copy the inner data
        auto full_data = Matrix<double>(height, width, length);
        for (int i = 0; i < field.nx; i++)
            for (int j = 0; j < field.ny; j++) 
              for (int k = 0; k < field.nz; k++) 
                 full_data(i, j, k) = field(i + 1, j + 1, k + 1);
          
#ifndef NO_MPI     
        // Receive data from other ranks
        for (int p = 1; p < parallel.size; p++) {
            int ix = parallel.coords[0] * field.nx;
            int iy = parallel.coords[1] * field.ny;
            int iz = parallel.coords[2] * field.nz;
            MPI_Recv(full_data.data(ix, iy, iz), 1, parallel.subarraytype, p, 22,
                     parallel.comm, MPI_STATUS_IGNORE);
        }
#endif
        // Write out the middle slice of data to a png file 
        std::ostringstream filename_stream;
        filename_stream << "heat_" << std::setw(4) << std::setfill('0') << iter << ".png";
        std::string filename = filename_stream.str();
#ifdef DISABLE_PNG
	std::cout << "No libpng, file not written" << std::endl;
#else
        save_png(full_data.data(height / 2, 0, 0), width, length, filename.c_str(), 'c');
#endif
    } else {
#ifndef NO_MPI     
        // Send data 
        MPI_Send(field.temperature.data(1, 1, 1), 1, parallel.subarraytype,
                 0, 22, parallel.comm);
#endif
    }

}

// Read the initial temperature distribution from a file
void read_field(Field& field, std::string filename,
                ParallelData& parallel)
{
    std::ifstream file;
    file.open(filename);
    // Read the header
    std::string line, comment;
    std::getline(file, line);
    int nx_full, ny_full;
    std::stringstream(line) >> comment >> nx_full >> ny_full;

    field.setup(nx_full, ny_full, ny_full, parallel);

    // Read the full array
    auto full = Matrix<double> (nx_full, ny_full, ny_full);

    if (0 == parallel.rank) {
        for (int i = 0; i < nx_full; i++)
            for (int j = 0; j < ny_full; j++)
                file >> full(i, j, 0);
    }

    file.close();

    // Inner region (no boundaries)
    auto inner = Matrix<double> (field.nx, field.ny, field.nz);

#ifndef NO_MPI
    MPI_Scatter(full.data(), field.nx * ny_full, MPI_DOUBLE, inner.data(),
                field.nx * ny_full, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    // Copy to the array containing also boundaries
    for (int i = 0; i < field.nx; i++)
        for (int j = 0; j < field.ny; j++)
             field(i + 1, j + 1, 0) = inner(i, j, 0);

    // Set the boundary values
    for (int i = 0; i < field.nx + 2; i++) {
        // left boundary
        field(i, 0, 0) = field(i, 1, 0);
        // right boundary
        field(i, field.ny + 1, 0) = field(i, field.ny, 0);
    }
    for (int j = 0; j < field.ny + 2; j++) {
        // top boundary
        field.temperature(0, j, 0) = field(1, j, 0);
        // bottom boundary
        field.temperature(field.nx + 1, j, 0) = field.temperature(field.nx, j, 0);
    }

}
