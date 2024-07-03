#include <gtest/gtest.h>
#include <vector>

/*
 * We want to creat a struct Field with some requirements:
 * 2D, with rows, cols
 * 2D indexing
 * usable on the cpu
 * usable on the gpu
 * */

template <typename T> struct Field {
    const size_t num_rows = 0ul;
    const size_t num_cols = 0ul;
    const T *data = nullptr;

    Field(size_t num_rows, size_t num_cols, const T *data)
        : num_rows(num_rows), num_cols(num_cols), data(data) {}

    static size_t getMemReq(size_t num_rows, size_t num_cols) {
        return sizeof(T) * num_rows * num_cols;
    }

    size_t operator()(size_t i, size_t j) const { return data[toLinear(i, j)]; }

    size_t toLinear(size_t i, size_t j) const { return i * num_cols + j; }

    T sum() const {
        T s = (T)0;
        for (size_t i = 0; i < num_rows * num_cols; i++) {
            s += data[i];
        }

        return s;
    }

    T avg() const { return sum() / numValues(); }

    size_t numValues() const { return num_cols * num_rows; }
};

TEST(field_test, construct_field) {
    constexpr size_t num_rows = 10;
    constexpr size_t num_cols = 10;
    const float *mem = nullptr;
    const Field field(num_rows, num_cols, mem);
}

TEST(field_test, get_mem_req_float) {
    constexpr size_t num_rows = 10;
    constexpr size_t num_cols = 10;
    const size_t num_bytes = Field<float>::getMemReq(num_rows, num_cols);
    ASSERT_EQ(num_bytes, sizeof(float) * num_rows * num_cols)
        << "Memory requirement incorrect";
}

TEST(field_test, get_mem_req_double) {
    constexpr size_t num_rows = 10;
    constexpr size_t num_cols = 10;
    const size_t num_bytes = Field<double>::getMemReq(num_rows, num_cols);
    ASSERT_EQ(num_bytes, sizeof(double) * num_rows * num_cols)
        << "Memory requirement incorrect";
}

TEST(field_test, indices_zero) {
    constexpr size_t num_rows = 10;
    constexpr size_t num_cols = 10;
    const size_t num_bytes = Field<float>::getMemReq(num_rows, num_cols);
    std::vector<float> data(num_bytes);
    const Field field(num_rows, num_cols, data.data());
    for (size_t row = 0; row < field.num_rows; row++) {
        for (size_t col = 0; col < field.num_cols; col++) {
            ASSERT_EQ(field(row, col), 0.0f) << "Field values non-zero";
        }
    }
}

TEST(field_test, indices_one) {
    constexpr size_t num_rows = 10;
    constexpr size_t num_cols = 10;
    const size_t num_bytes = Field<float>::getMemReq(num_rows, num_cols);
    std::vector<float> data(num_bytes, 1.0f);
    const Field field(num_rows, num_cols, data.data());
    for (size_t row = 0; row < field.num_rows; row++) {
        for (size_t col = 0; col < field.num_cols; col++) {
            ASSERT_EQ(field(row, col), 1.0f) << "Field values not one";
        }
    }
}

TEST(field_test, zero_zero_is_zero) {
    const Field<float> f(10, 10, nullptr);
    const auto i = f.toLinear(0, 0);
    ASSERT_EQ(0, i) << "(0, 0) should be 0";
}

TEST(field_test, zero_one_is_one) {
    const Field<float> f(10, 10, nullptr);
    const auto i = f.toLinear(0, 1);
    ASSERT_EQ(1, i) << "(0, 1) should be 1";
}

TEST(field_test, zero_field_sum_is_zero) {
    constexpr size_t num_rows = 10;
    constexpr size_t num_cols = 10;
    const size_t num_bytes = Field<float>::getMemReq(num_rows, num_cols);
    std::vector<float> data(num_bytes);
    const Field field(num_rows, num_cols, data.data());
    ASSERT_EQ(field.sum(), 0.0f) << "Zero field sum is not zero";
}

TEST(field_test, unity_field_sum_is_num_elements) {
    constexpr size_t num_rows = 10;
    constexpr size_t num_cols = 10;
    const size_t num_bytes = Field<float>::getMemReq(num_rows, num_cols);
    std::vector<float> data(num_bytes, 1.0f);
    const Field field(num_rows, num_cols, data.data());
    ASSERT_EQ(field.sum(), (float)(field.numValues()))
        << "Unity field sum is not num elements";
}

TEST(field_test, unity_field_avg_is_one) {
    constexpr size_t num_rows = 10;
    constexpr size_t num_cols = 10;
    const size_t num_bytes = Field<float>::getMemReq(num_rows, num_cols);
    std::vector<float> data(num_bytes, 1.0f);
    const Field field(num_rows, num_cols, data.data());
    ASSERT_EQ(field.avg(), 1.0f) << "Unity field average is not one";
}

TEST(field_test, num_elements_is_num_rows_times_num_cols) {
    constexpr size_t num_rows = 10;
    constexpr size_t num_cols = 10;
    const Field<float> field(num_rows, num_cols, nullptr);
    ASSERT_EQ(field.numValues(), num_rows * num_cols)
        << "Num values is not num_rows x num_cols";
}
