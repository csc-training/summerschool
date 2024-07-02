#include <gtest/gtest.h>
#include "io.hpp"

TEST(io_test, make_png_filename_0) {
    ASSERT_STREQ("heat_0000.png", heat::make_png_filename("heat_", 0).c_str());
}

TEST(io_test, make_png_filename_500) {
    ASSERT_STREQ("heat_0500.png",
                 heat::make_png_filename("heat_", 500).c_str());
}

TEST(io_test, make_png_filename_9999) {
    ASSERT_STREQ("heat_9999.png",
                 heat::make_png_filename("heat_", 9999).c_str());
}
