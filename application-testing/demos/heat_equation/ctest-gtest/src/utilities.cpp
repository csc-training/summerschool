// Utility functions for heat equation solver

#include <mpi.h>
#include <tuple>

#include "field.hpp"
#include "parallel.hpp"
#include "utilities.hpp"

namespace heat {
double average(const Field &field, const ParallelData &pd) {
    // Divide the global sum by the global size
    return sum(field.sum()) / (field.num_rows * field.num_cols * pd.size);
}

double sum(double local_sum) {
    // Compute global sum by summing the local sums
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    return global_sum;
}

std::tuple<int, int, std::vector<double>> generate_field(int num_rows,
                                                         int num_cols) {
    /* Generate a temperature field with a cold disk in a hot soup
     * H H H H H H H H H H H H H
     * H H H H H H H H H H H H H
     * H H H H H H H H H  HH H H
     * H H H H H C C C H H H H H
     * H H H H C C C C C H H H H
     * H H H H C C C C C H H H H
     * H H H H C C C C C H H H H
     * H H H H H C C C H H H H H
     * H H H H H H H H H H H H H
     * H H H H H H H H H H H H H
     * H H H H H H H H H H H H H
     */
    std::vector<double> data;
    data.reserve(num_rows * num_cols);
    // Radius of the source disc
    const auto radius = num_rows / 6.0;
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            // Distance of point i, j from the origin
            const auto dx = i - num_rows / 2;
            const auto dy = j - num_cols / 2;
            auto temperature = dx * dx + dy * dy < radius * radius ? 5.0 : 65.0;
            data.push_back(temperature);
        }
    }
    return std::make_tuple(num_rows, num_cols, data);
}

std::vector<double> scatter(std::vector<double> &&full_data,
                            int num_values_per_rank) {
    // Scatter data to MPI processes
    std::vector<double> my_data(num_values_per_rank);
    MPI_Scatter(full_data.data(), num_values_per_rank, MPI_DOUBLE,
                my_data.data(), num_values_per_rank, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    return my_data;
}

std::vector<double> gather(std::vector<double> &&my_data,
                           int num_total_values) {
    // Gather data from MPI procesess to a single vector
    std::vector<double> full_data(num_total_values);
    MPI_Gather(my_data.data(), my_data.size(), MPI_DOUBLE, full_data.data(),
               my_data.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return full_data;
}
} // namespace heat
