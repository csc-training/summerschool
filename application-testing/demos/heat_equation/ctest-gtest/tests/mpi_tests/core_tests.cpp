#include <gtest/gtest.h>
#include <numeric>

#include "field.hpp"
#include "parallel.hpp"
#include "core.hpp"

TEST(core_test, exchange_data) {
    constexpr int num_rows = 1600;
    constexpr int num_cols = 64;
    constexpr int n = num_rows * num_cols;
    heat::ParallelData pd;
    std::vector<double> v(num_rows * num_cols);
    const auto first = pd.rank * n + 1;
    std::iota(v.begin(), v.end(), first);
    heat::Field field(std::move(v), num_rows, num_cols);

    heat::exchange(field, pd);

    const auto last_of_prev = first - num_cols * (!!pd.rank);
    const auto first_of_next = first + n - num_cols * (pd.rank == pd.size - 1);
    for (int j = 0; j < num_cols; j++) {
        // Ghost layer above my values
        ASSERT_EQ(field(-1, j), last_of_prev + j);
        // Ghost layer below my values
        ASSERT_EQ(field(num_rows, j), first_of_next + j);
    }
}
