#pragma once
#include <string>
#include <vector>
#include <mpi.h>

// Class for basic parallelization information
struct ParallelData {
    int size;            // Number of MPI tasks
    int rank;
    int nup, ndown;      // Ranks of neighbouring MPI tasks

    ParallelData() {      // Constructor

      MPI_Comm_size(MPI_COMM_WORLD, &size);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      nup = rank - 1;
      ndown = rank + 1;

      if (nup < 0) {
        nup = MPI_PROC_NULL;
      }
      if (ndown > size - 1) {
        ndown = MPI_PROC_NULL;
      }
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

    std::vector<double> temperature;

    void setup(int nx_in, int ny_in, ParallelData parallel);

    void generate(ParallelData parallel);

};

// Function declarations
void initialize(int argc, char *argv[], Field& current,
                Field& previous, int& nsteps, ParallelData parallel);

void exchange(Field& field, const ParallelData parallel);

void evolve(Field& curr, Field& prev, const double a, const double dt);

void write_field(const Field& field, const int iter, const ParallelData parallel);

void read_field(Field& field, std::string filename,
                ParallelData parallel);

double average(const Field& field, const ParallelData parallel);

void enter_data(Field& temperature1, Field& temperature2);

void exit_data(Field& temperature1, Field& temperature2);

void update_host(Field& temperature);

void update_device(Field& temperature);

