#include <gtest/gtest.h>

#include "io.hpp"

TEST(io_test, default_input_ok) {
    const heat::Input default_input = {};
    const heat::Input input = heat::read_input("", 0);
    ASSERT_EQ(input, default_input) << "input is different from default_input";
}

TEST(io_test, input_from_file_ok) {
    const heat::Input input = heat::read_input("testdata/input.json", 0);
    const heat::Input default_input = {};
    ASSERT_NE(input, default_input) << "input is equal to default_input";
}

TEST(io_test, input_from_nonexistent_path_throws_exception) {
    EXPECT_THROW(
        {
            try {
                const heat::Input input =
                    heat::read_input("batman vs superman", 0);
            } catch (const std::runtime_error &e) {
                EXPECT_STREQ("Non-existent path: \"batman vs superman\"",
                             e.what());
                throw;
            }
        },
        std::runtime_error);
}

TEST(io_test, read_field_data_from_file) {
    auto [num_rows, num_cols, data] = heat::read_field("testdata/bottle.dat");
    ASSERT_EQ(data.size(), 40000);
    ASSERT_EQ(num_rows * num_cols, data.size());
}
