#pragma once

#include <vector>

namespace heat {
struct Field;
struct ParallelData;
// Global average of the field
double average(const Field &field, const ParallelData &pd);
// Global sum of the field
double sum(double local_sum);
// Field generation
std::tuple<int, int, std::vector<double>> generate_field(int num_rows,
                                                         int num_cols);
// Scatter data among MPI processes
std::vector<double> scatter(std::vector<double> &&full_data,
                            int num_values_per_rank);
// Gather data from MPI processes
std::vector<double> gather(std::vector<double> &&my_data, int num_total_values);
} // namespace heat
