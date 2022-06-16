#pragma once
#include <string>
#include "matrix.hpp"
#include <mpi.h>

// Class for basic parallelization information
struct ParallelData {
    int size;            // Number of MPI tasks
    int rank;
    int nghbrs[2][2]; // Ranks of neighbouring MPI tasks
    MPI_Datatype rowtype, columntype, subarraytype;
    MPI_Comm comm;

    ParallelData() {      // Constructor

      MPI_Comm_size(MPI_COMM_WORLD, &size);

      constexpr int ndims = 2;
      int dims[ndims] = {0, 0};
      int periods[ndims] = {0, 0};

      MPI_Dims_create(size, ndims, dims);
      MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1, &comm);
      MPI_Comm_rank(comm, &rank);

      // Determine neighbors
      for (int i=0; i < ndims; i++)
        MPI_Cart_shift(comm, i, 1, &nghbrs[i][0], &nghbrs[i][1]);
    };

};

// Class for temperature field
struct Field {
    // nx and ny are the true dimensions of the field. The temperature matrix
    // contains also ghost layers, so it will have dimensions nx+2 x ny+2 
    int nx;                     // Local dimensions of the field
    int ny;
    int nx_full;                // Global dimensions of the field
    int ny_full;                // Global dimensions of the field
    double dx = 0.01;           // Grid spacing
    double dy = 0.01;

    Matrix<double> temperature;

    void setup(int nx_in, int ny_in, ParallelData& parallel);

    void generate(ParallelData parallel);

    // standard (i,j) syntax for setting elements
    double& operator()(int i, int j) {return temperature(i, j);}

    // standard (i,j) syntax for getting elements
    const double& operator()(int i, int j) const {return temperature(i, j);}

};

// Function declarations
void initialize(int argc, char *argv[], Field& current,
                Field& previous, int& nsteps, ParallelData& parallel);

void exchange(Field& field, const ParallelData parallel);

void evolve(Field& curr, const Field& prev, const double a, const double dt);

void write_field(Field& field, const int iter, const ParallelData parallel);

void read_field(Field& field, std::string filename,
                ParallelData& parallel);

double average(const Field& field, const ParallelData parallel);
