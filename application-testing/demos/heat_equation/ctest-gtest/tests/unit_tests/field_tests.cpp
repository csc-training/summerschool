#include "constants.hpp"
#include "field.hpp"
#include "io.hpp"

#include <gtest/gtest.h>
#include <numeric>

TEST(field_test, domain_partition_succeeds) {
    constexpr int num_rows = 100;
    constexpr int num_cols = 100;
    constexpr int num_partitions = 10;
    EXPECT_NO_THROW(
        heat::Field::partition_domain(num_rows, num_cols, num_partitions));
}

TEST(field_test, domain_partition_throws_an_exception) {
    constexpr int num_rows = 101;
    constexpr int num_cols = 100;
    constexpr int num_partitions = 10;
    EXPECT_THROW(
        {
            try {
                heat::Field::partition_domain(num_rows, num_cols,
                                              num_partitions);
            } catch (const std::runtime_error &e) {
                EXPECT_STREQ("Could not partition 101 rows and 100 columns "
                             "evenly to 10 partitions",
                             e.what());
                throw;
            }
        },
        std::runtime_error);
}

TEST(field_test, field_construction) {
    constexpr int num_rows = 2000;
    constexpr int num_cols = 2000;
    std::vector<double> v(num_rows * num_cols);
    std::iota(v.begin(), v.end(), 0.0);
    const heat::Field field(std::move(v), num_rows, num_cols);

    ASSERT_EQ(field.num_rows, num_rows);
    ASSERT_EQ(field.num_cols, num_cols);

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            const double value = static_cast<double>(i * num_cols + j);
            // Field doesn't sample the ghost layers
            ASSERT_EQ(field(i, j), value);
        }
    }
}

TEST(field_test, zero_field_sum_is_zero) {
    constexpr int num_rows = 2000;
    constexpr int num_cols = 2000;
    const heat::Field field(std::vector<double>(num_rows * num_cols), num_rows,
                            num_cols);

    ASSERT_EQ(field.sum(), 0.0);
}

TEST(field_test, unity_field_sum_is_num_items) {
    constexpr int num_rows = 2000;
    constexpr int num_cols = 2000;
    const heat::Field field(std::vector<double>(num_rows * num_cols, 1.0),
                            num_rows, num_cols);

    ASSERT_EQ(field.sum(), static_cast<double>(num_rows * num_cols));
}

TEST(field_test, get_data_yields_correct_data) {
    constexpr int num_rows = 600;
    constexpr int num_cols = 240;
    std::vector<double> v(num_rows * num_cols);
    std::iota(v.begin(), v.end(), 0.0);
    const heat::Field field(std::move(v), num_rows, num_cols);
    const auto data = field.get_temperatures();
    ASSERT_EQ(data.size(), num_rows * num_cols);
    double value = 0;
    for (const auto &item : data) {
        ASSERT_EQ(item, value++);
    }
}

TEST(field_test, swap_data) {
    constexpr int num_rows = 600;
    constexpr int num_cols = 240;

    std::vector<double> v(num_rows * num_cols);
    std::iota(v.begin(), v.end(), 0.0);
    heat::Field f(std::move(v), num_rows, num_cols);
    heat::Field g(std::vector<double>(num_rows * num_cols), num_rows, num_cols);
    f.swap(g);

    double value = 0.0;
    for (int i = 0; i < f.num_rows; i++) {
        for (int j = 0; j < f.num_cols; j++) {
            ASSERT_EQ(f(i, j), 0.0);
            ASSERT_EQ(g(i, j), value++);
        }
    }
}

TEST(field_test, to_up_correct) {
    constexpr int num_rows = 600;
    constexpr int num_cols = 240;
    std::vector<double> v(num_rows * num_cols);
    std::iota(v.begin(), v.end(), 0.0);
    heat::Field field(std::move(v), num_rows, num_cols);
    auto to_up = field.to_up();
    ASSERT_EQ(*to_up, 0.0);
    ASSERT_EQ(*(to_up + 1), 1.0);
    ASSERT_EQ(*(to_up - 1), 0.0);
    ASSERT_EQ(*(to_up - 2), static_cast<double>(num_cols - 1));
    ASSERT_EQ(*(to_up - 3), static_cast<double>(num_cols - 1));
}

TEST(field_test, to_down_correct) {
    constexpr int num_rows = 200;
    constexpr int num_cols = 40;
    std::vector<double> v(num_rows * num_cols);
    std::iota(v.begin(), v.end(), 0.0);
    heat::Field field(std::move(v), num_rows, num_cols);
    auto to_down = field.to_down();
    constexpr auto value = (num_rows - 1) * num_cols;
    ASSERT_EQ(*to_down, value);
    ASSERT_EQ(*(to_down + 1), value + 1.0);
    ASSERT_EQ(*(to_down - 1), value);
    ASSERT_EQ(*(to_down + num_cols + 1), value);
    ASSERT_EQ(*(to_down + num_cols + 2), value);
    ASSERT_EQ(*(to_down + num_cols + 3), value + 1);
}

TEST(field_test, from_up_correct) {
    constexpr int num_rows = 60;
    constexpr int num_cols = 20;
    std::vector<double> v(num_rows * num_cols);
    std::iota(v.begin(), v.end(), 0.0);
    heat::Field field(std::move(v), num_rows, num_cols);
    ASSERT_EQ(field.from_up() + num_cols + 2, field.to_up());
}

TEST(field_test, from_down_correct) {
    constexpr int num_rows = 60;
    constexpr int num_cols = 20;
    std::vector<double> v(num_rows * num_cols);
    std::iota(v.begin(), v.end(), 0.0);
    heat::Field field(std::move(v), num_rows, num_cols);
    ASSERT_EQ(field.from_down() - num_cols - 2, field.to_down());
}

TEST(field_test, sample_constant_field_correctly) {
    heat::Input input;
    heat::Constants constants(input);
    constexpr int num_rows = 60;
    constexpr int num_cols = 20;
    std::vector<double> v(num_rows * num_cols, 6.66f);
    const heat::Field field(std::move(v), num_rows, num_cols);
    constexpr int i = num_rows / 3;
    constexpr int j = num_cols / 4;
    ASSERT_EQ(field.sample(i, j, constants), field(i, j));
}
