#include <gtest/gtest.h>
#include <numeric>

#include "field.hpp"
#include "parallel.hpp"
#include "utilities.hpp"

TEST(utilities_test, zero_field_average_is_zero) {
    constexpr int num_rows = 2000;
    constexpr int num_cols = 2000;
    const heat::Field field(std::vector<double>(num_rows * num_cols), num_rows,
                            num_cols);
    heat::ParallelData pd;
    ASSERT_EQ(heat::average(field, pd), 0.0);
}

TEST(utilities_test, unity_field_average_is_one) {
    constexpr int num_rows = 2000;
    constexpr int num_cols = 100;
    const heat::Field field(std::vector<double>(num_rows * num_cols, 1.0),
                            num_rows, num_cols);
    heat::ParallelData pd;
    ASSERT_EQ(heat::average(field, pd), 1.0);
}

TEST(utilities_test, iota_field_average_correct) {
    constexpr int num_rows = 1616;
    constexpr int num_cols = 512;
    constexpr int n = num_rows * num_cols;
    heat::ParallelData pd;
    std::vector<double> v(num_rows * num_cols);
    std::iota(v.begin(), v.end(), pd.rank * n + 1);
    const heat::Field field(std::move(v), num_rows, num_cols);
    // 1 + 2 + 3 + ... + n = (n * (n + 1)) / 2
    double sum = static_cast<double>(pd.size * n);
    sum *= (sum + 1.0);
    sum /= 2.0;
    ASSERT_EQ(heat::average(field, pd), sum / (pd.size * n));
}

TEST(utilities_test, scatter_successfully) {
    constexpr int num_rows = 1616;
    constexpr int num_cols = 512;
    constexpr int n = num_rows * num_cols;
    heat::ParallelData pd;
    const int n_per_rank = n / pd.size;
    std::vector<double> full_data;
    if (pd.rank == 0) {
        full_data.resize(n);
        std::iota(full_data.begin(), full_data.end(), 0);
    }
    const auto my_data = heat::scatter(std::move(full_data), n / pd.size);

    double i = static_cast<double>(pd.rank * n_per_rank);
    for (const auto &item : my_data) {
        ASSERT_EQ(item, i++);
    }
}

TEST(utilities_test, gather_successfully) {
    constexpr int num_rows = 160;
    constexpr int num_cols = 20;
    constexpr int n = num_rows * num_cols;
    heat::ParallelData pd;
    std::vector<double> v(num_rows * num_cols);
    std::iota(v.begin(), v.end(), pd.rank * n);

    const auto full_data = heat::gather(std::move(v), n * pd.size);

    if (pd.rank == 0) {
        ASSERT_EQ(full_data.size(), n * pd.size);
        double i = 0.0;
        for (const auto &item : full_data) {
            ASSERT_EQ(item, i++);
        }
    }
}

TEST(utilities_test, global_sum_correct1) {
    heat::ParallelData pd;
    ASSERT_EQ(heat::sum(1.0), static_cast<double>(pd.size));
}

TEST(utilities_test, global_sum_correct2) { ASSERT_EQ(heat::sum(0.0), 0.0); }
