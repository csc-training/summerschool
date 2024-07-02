#include "field.hpp"
#include "constants.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <mpi.h>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace heat {
Field::Field(std::vector<double> &&data, int num_rows, int num_cols)
    : num_rows(num_rows), num_cols(num_cols),
      temperatures((num_rows + 2) * (num_cols + 2)) {
    // Copy the real data to the inner part of the temperature field
    for (int i = 0; i < num_rows; i++) {
        const int row = i + 1;
        const int width = num_cols + 2;
        constexpr int column = 1;
        const int offset = row * width + column;
        auto from = data.begin() + i * num_cols;
        auto to = temperatures.begin() + offset;
        std::copy_n(from, num_cols, to);
    }

    // Make the ghost layers by copying the value from the closest real
    // row/column
    const int nr = num_rows + 2;
    const int nc = num_cols + 2;
    for (int i = 0; i < nr; i++) {
        const int first = i * nc;
        const int last = (i + 1) * nc - 1;
        // Left
        temperatures[first] = temperatures[first + 1];
        // Right
        temperatures[last] = temperatures[last - 1];
    }

    for (int j = 0; j < nc; j++) {
        const int first = j;
        const int last = (nr - 1) * nc + j;
        // Top
        temperatures[first] = temperatures[first + nc];
        // Bottom
        temperatures[last] = temperatures[last - nc];
    }
}

double Field::sum() const {
    // Sum the real values of the field
    double sum = 0.0;
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            sum += (*this)(i, j);
        }
    }

    return sum;
}

std::vector<double> Field::get_temperatures() const {
    // Copy the real data of the field, skipping the ghost layers
    std::vector<double> data;
    data.reserve(num_rows * num_cols);

    for (int i = 0; i < num_rows; i++) {
        const int row = i + 1;
        const int width = num_cols + 2;
        constexpr int column = 1;
        const int offset = row * width + column;
        auto from = temperatures.begin() + offset;
        std::copy_n(from, num_cols, std::back_inserter(data));
    }

    return data;
}

double Field::sample(int i, int j, const Constants &constants) const {
    /* The five point stencil sampling
     * - - - - -
     * - - + - -
     * - + + + -
     * - - + - -
     * - - - - -
     * */
    const auto center = (*this)(i, j);
    const auto up = (*this)(i - 1, j);
    const auto down = (*this)(i + 1, j);
    const auto left = (*this)(i, j - 1);
    const auto right = (*this)(i, j + 1);
    const auto x = right + left - 2.0 * center;
    const auto y = down + up - 2.0 * center;
    return center + constants.a * constants.dt *
                        (x * constants.inv_dx2 + y * constants.inv_dy2);
}

std::pair<int, int> Field::partition_domain(int num_rows, int num_cols,
                                            int num_partitions) {
    /* Attempt to partition the domain evenly to all processes
     * The columns are not partitioned, i.e. each process has a certain number
     * of full rows.
     * N.B. This function doesn't handle any data, it only computes the
     * partition sizes
     *
     * Example:
     * 12x13 matrix is divided to four processes:
     *    0 0 0 0 0 0 0 0 0 0 1 1 1
     *    0 1 2 3 4 5 6 7 8 9 0 1 2
     * 00 D D D D D D D D D D D D D \
     * 01 D D D D D D D D D D D D D  -> data for process 0
     * 02 D D D D D D D D D D D D D /
     * 03 D D D D D D D D D D D D D \
     * 04 D D D D D D D D D D D D D  -> data for process 1
     * 05 D D D D D D D D D D D D D /
     * 06 D D D D D D D D D D D D D \
     * 07 D D D D D D D D D D D D D  -> data for process 2
     * 08 D D D D D D D D D D D D D /
     * 09 D D D D D D D D D D D D D \
     * 10 D D D D D D D D D D D D D  -> data for process 3
     * 11 D D D D D D D D D D D D D /
     */
    const int nr = num_rows / num_partitions;
    if (nr * num_partitions != num_rows) {
        std::stringstream ss;
        ss << "Could not partition " << num_rows << " rows and " << num_cols
           << " columns evenly to " << num_partitions << " partitions";
        throw std::runtime_error(ss.str());
    }
    // Columns are not partitioned
    const int nc = num_cols;

    return std::make_pair(nr, nc);
}
} // namespace heat
