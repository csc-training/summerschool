#pragma once
#include <string>
#include <vector>

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

    void setup(int nx_in, int ny_in);

    void generate();

};

// Function declarations
void initialize(int argc, char *argv[], Field& current,
                Field& previous, int& nsteps);

void evolve(Field& curr, Field& prev, const double a, const double dt);

void write_field(const Field& field, const int iter);

void read_field(Field& field, std::string filename);

double average(const Field& field);

void enter_data(Field& temperature1, Field& temperature2);

void exit_data(Field& temperature1, Field& temperature2);

void update_host(Field& temperature);

