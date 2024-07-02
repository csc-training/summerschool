#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>

#include "parallel.hpp"
#include "pngwriter.h"

namespace heat {
void run(std::string &&);
}

struct PngData {
    int nx = 0;
    int ny = 0;
    int channels = 0;
    uint8_t *data = nullptr;

    ~PngData() {
        if (data != nullptr) {
            release_png(data);
        }
    }
};

bool loadPng(const char *fname, PngData &png) {
    auto path = (std::filesystem::current_path() / fname);

    if (std::filesystem::exists(path)) {
        path = std::filesystem::is_symlink(path)
                   ? std::filesystem::read_symlink(path)
                   : path;

        const auto filename = path.c_str();
        png.data = load_png(filename, &png.nx, &png.ny, &png.channels);

        return true;
    }

    return false;
}

TEST(integration_test, image_matches_reference_with_input) {
    heat::run("testdata/input.json");

    PngData reference_data = {};
    ASSERT_TRUE(loadPng("testdata/bottle_0500.png", reference_data))
        << "Could not load reference data from png";

    PngData computed_data = {};
    ASSERT_TRUE(loadPng("heat_0500.png", computed_data))
        << "Could not load computed data from png";

    ASSERT_EQ(reference_data.nx, computed_data.nx);
    ASSERT_EQ(reference_data.ny, computed_data.ny);
    ASSERT_EQ(reference_data.channels, computed_data.channels);

    const int num_bytes =
        reference_data.nx * reference_data.ny * reference_data.channels;
    for (int i = 0; i < num_bytes; i++) {
        ASSERT_EQ(reference_data.data[i], computed_data.data[i])
            << "Computed data differs from reference data at byte " << i
            << "\nReference: " << reference_data.data[i]
            << ", computed: " << computed_data.data[i];
    }
}

TEST(integration_test, image_matches_reference_with_defaults) {
    heat::run("");

    PngData reference_data = {};
    ASSERT_TRUE(loadPng("testdata/default_0500.png", reference_data))
        << "Could not load reference data from png";

    PngData computed_data = {};
    ASSERT_TRUE(loadPng("heat_0500.png", computed_data))
        << "Could not load computed data from png";

    ASSERT_EQ(reference_data.nx, computed_data.nx);
    ASSERT_EQ(reference_data.ny, computed_data.ny);
    ASSERT_EQ(reference_data.channels, computed_data.channels);

    const int num_bytes =
        reference_data.nx * reference_data.ny * reference_data.channels;
    for (int i = 0; i < num_bytes; i++) {
        ASSERT_EQ(reference_data.data[i], computed_data.data[i])
            << "Computed data differs from reference data at byte " << i
            << "\nReference: " << reference_data.data[i]
            << ", computed: " << computed_data.data[i];
    }
}
