#pragma once
#include "heat.hpp"
#include "parallel.hpp"
#include <hip/hip_runtime_api.h>

// Function declarations
void initialize(int argc, char *argv[], Field& current,
                Field& previous, int& nsteps, ParallelData& parallel);

void exchange_init(Field& field, ParallelData& parallel);

void exchange_finalize(Field& field, ParallelData& parallel);

void evolve(Field& curr, Field& prev, const double a, const double dt);

void evolve_interior(Field& curr, Field& prev, const double a, const double dt, hipStream_t *streams);

void evolve_edges(Field& curr, Field& prev, const double a, const double dt, hipStream_t *streams);

void write_field(Field& field, const int iter, const ParallelData& parallel);

void read_field(Field& field, std::string filename,
                ParallelData& parallel);

double average(const Field& field);

double timer();

void exit_data(Field& curr, Field& prev);

void enter_data(Field& curr, Field& prev);

void update_host(Field& temperature);

void update_device(Field& temperature);

void allocate_data(Field& field1, Field& field2);

void free_data(Field& field1, Field& field2);

