#include <fstream>
#include <hip/hip_runtime.h>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <vector>

#define HIP_ERRCHK(result) hip_errchk(result, __FILE__, __LINE__)
static inline void hip_errchk(hipError_t result, const char *file, int line) {
    if (result != hipSuccess) {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(result), file,
               line);
        exit(EXIT_FAILURE);
    }
}

// Copy all elements using threads in a 2D grid
__global__ void copy2d(double *dst, double *src, size_t num_cols,
                       size_t num_rows) {
    const size_t col_start = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t row_start = threadIdx.y + blockIdx.y * blockDim.y;
    const size_t col_stride = blockDim.x * gridDim.x;
    const size_t row_stride = blockDim.y * gridDim.y;

    for (size_t row = row_start; row < num_rows; row += row_stride) {
        for (size_t col = col_start; col < num_cols; col += col_stride) {
            const size_t index = row * num_cols + col;
            dst[index] = src[index];
        }
    }

    // clang-format off
    /*
     * In this ASCII art example we have a grid of
     * - 2x2 blocks of
     * - 4x4 threads
     * - going over a 10 x 10 2D array:
     *
     *     ______col_stride_______
     *    |                       |
     *   _|_______________________|_________
     *  | 00 01 02 03 | 04 05 06 07 | 08 09 |---
     *  | 10 11 12 13 | 14 15 16 17 | 18 19 |   |
     *  | 20 21 22 23 | 24 25 26 27 | 28 29 |   |
     *  |_30_31_32_33_|_34_35_36_37_|_38_39_|   |
     *  | 40 41 42 43 | 44 45 46 47 | 48 49 |   | row_stride
     *  | 50 51 52 53 | 54 55 56 57 | 58 59 |   |
     *  | 60 61 62 63 | 64 65 66 67 | 68 69 |   |
     *  |_70_71_72_73_|_74_75_76_77_|_78_79_|---
     *  | 80 81 82 83 | 84 85 86 87 | 88 89 |
     *  |_90_91_92_93_| 94_95_96_97_|_98_99_|
     *
     *
     *    Which block processes which area?
     *   ___________________________________
     *  |             |             |       |
     *  |   (0, 0)    |   (1, 0)    | (0, 0)|
     *  |             |             |       |
     *  |____________ |_____________|_______|
     *  |             |             |       |
     *  |   (0, 1)    |   (1, 1)    | (0, 1)|
     *  |             |             |       |
     *  |_____________|_____________|_______|
     *  |   (0, 0)    |   (1, 0)    | (0, 0)|
     *  |_____________| ____________|_______|
     *
     *
     * # The first iteration of the outer loop (rows)
     * ## The first iteration of the inner loop (columns)
     *
     * | blockIdx (x, y) | processed values           |
     * |-----------------|----------------------------|
     * |          (0, 0) | 00-03, 10-13, 20-23, 30-33 |
     * |          (1, 0) | 04-07, 14-17, 24-27, 34-37 |
     * |          (0, 1) | 40-43, 50-53, 60-63, 70-73 |
     * |          (1, 1) | 44-47, 54-57, 64-67, 74-77 |
     *
     * ## The second iteration of the inner loop (columns)
     * block (0, 0) will add col_stride to its columns and its threads will process
     * the remaining eight values in the top right area:
     *
     * | thread (x, y) | processed value |
     * |---------------|-----------------|
     * |        (0, 0) |              08 |
     * |        (1, 0) |              09 |
     * |        (0, 1) |              18 |
     * |        (1, 1) |              19 |
     * |        (0, 2) |              28 |
     * |        (1, 2) |              29 |
     * |        (0, 3) |              38 |
     * |        (1, 3) |              39 |
     *
     * The 8 remaining threads of block (0, 0) wait for the other 8 to finish.
     *
     * Similarly, block (0, 1) adds col_stride to its columns and its 8 threads process
     * the center right area of 8 values.
     *
     * Blocks (1, 0) and (1, 1) add col_stride to their columns,
     * but those are outside the 10x10 array.
     *
     * # The second iteration of the outer loop (rows)
     * ## The first iteration of the inner loop (columns)
     * Every block adds row_stride to their rows.
     *
     * block (0, 0) will process the 8 values on the bottom left:
     * | thread (x, y) | processed value |
     * |---------------|-----------------|
     * |        (0, 0) |              80 |
     * |        (1, 0) |              81 |
     * |        (2, 0) |              82 |
     * |        (3, 0) |              83 |
     * |        (0, 1) |              90 |
     * |        (1, 1) |              91 |
     * |        (2, 1) |              92 |
     * |        (3, 1) |              93 |
     *
     * Similarly block (1, 0) will process the 8 values on the bottom center.
     * Blocks (0, 1) and (1, 1) are outside the 10x10 array, so they do nothing.
     *
     * ## The second iteration of the inner loop (columns)
     * Blocks (0, 0) and (1, 0) add col_stride to their columns.
     * Block (0, 0) will process the remaining four values at the bottom right corner:
     *
     * | thread (x, y) | processed value |
     * |---------------|-----------------|
     * |        (0, 0) |              88 |
     * |        (1, 0) |              89 |
     * |        (0, 1) |              98 |
     * |        (1, 1) |              99 |
     *
     * Block (1, 0) is outside the 10x10 array and does nothing.
     *
     * Now every value of the 10x10 array has been processed.
     * --------------------------------------------------------------------------------
     *
     * Do note, that with this artificially small problem the work is quite unbalanced:
     * | block (x, y) | #areas processed |
     * |--------------|------------------|
     * |       (0, 0) |                4 |
     * |       (1, 0) |                2 |
     * |       (0, 1) |                2 |
     * |       (1, 1) |                1 |
     *
     * But in real problems the data sizes are usually much, much larger and the ratio
     * of data elements to threads in grid is also much larger, thus yielding a better balance.
     *
     * The looped 2D grid "extends to infinity", i.e. one can repeat the
     * structure of the 2D grid over and over again in columns and in rows
     * to cover an arbitrary large/small area.
     *
     * Below, the 2D grid is repeated three times in the x direction and twice in the y direction.
     * 'a' marks the extent of a 2D array that would produce this kind of repetition.
     * The values (x, y) inside the boxes tell the blockIdx of the block processing the area.
     * Note that changing the number of threads in a block or the number of blocks in a grid
     * changes this configuration.
     *
     *
     *    first iteration of cols     second iteration of cols    third iteration of cols    
     *  |<------------------------->|<------------------------->|<------------------------->|
     *  |___________________________|___________________________|___________________________|__first_iteration of rows
     *  |             |             |             |             |          a  |             |             ^
     *  |   (0, 0)    |   (1, 0)    |   (0, 0)    |   (1, 0)    |   (0, 0) a  |   (1, 0)    |             |
     *  |             |             |             |             |          a  |             |             |
     *  |_____________|_____________|_____________|_____________|__________a__|_____________|             |
     *  |             |             |             |             |          a  |             |             |
     *  |   (0, 1)    |   (1, 1)    |   (0, 1)    |   (1, 1)    |   (0, 1) a  |   (1, 1)    |             |
     *  |             |             |             |             |          a  |             |             v
     *  |_____________|_____________|_____________|_____________|__________a__|_____________|__second_iteration of rows
     *  |             |             |             |             |          a  |             |             ^
     *  |   (0, 0)    |   (1, 0)    |   (0, 0)    |   (1, 0)    |   (0, 0) a  |   (1, 0)    |             |
     *  |             |             |             |             |          a  |             |             |
     *  |_____________|_____________|_____________|_____________|__________a__|_____________|             |
     *  |             |             |             |             |          a  |             |             |
     *  |   (0, 1)    |   (1, 1)    |   (0, 1)    |   (1, 1)    |   (0, 1) a  |   (1, 1)    |             |
     *  |aaaaaaaaaaaaa|aaaaaaaaaaaaa|aaaaaaaaaaaaa|aaaaaaaaaaaaa|aaaaaaaaaaa  |             |             |
     *  |_____________|_____________|_____________|_____________|_____________|_____________|_____________v
     *
     */
    // clang-format on
}

namespace index_visualisation { void output(); }

int main() {
    static constexpr size_t num_cols = 600;
    static constexpr size_t num_rows = 400;
    static constexpr size_t num_values = num_cols * num_rows;
    static constexpr size_t num_bytes = sizeof(double) * num_values;
    std::vector<double> x(num_values);
    std::vector<double> y(num_values, 0.0);

    // Initialise data
    for (size_t i = 0; i < num_values; i++) {
        x[i] = static_cast<double>(i) / 1000.0;
    }

    void *d_x = nullptr;
    void *d_y = nullptr;
    // Allocate + copy initial values
    HIP_ERRCHK(hipMalloc(&d_x, num_bytes));
    HIP_ERRCHK(hipMalloc(&d_y, num_bytes));
    HIP_ERRCHK(hipMemcpy(d_x, static_cast<void *>(x.data()), num_bytes,
                         hipMemcpyDefault));
    HIP_ERRCHK(hipMemcpy(d_y, static_cast<void *>(y.data()), num_bytes,
                         hipMemcpyDefault));

    // Define grid dimensions + launch the device kernel
    const dim3 threads(64, 16, 1);
    const dim3 blocks(64, 64, 1);

    copy2d<<<blocks, threads>>>(static_cast<double *>(d_y),
                                static_cast<double *>(d_x), num_cols, num_rows);

    // Copy results back to CPU
    HIP_ERRCHK(hipMemcpy(static_cast<void *>(y.data()), d_y, num_bytes,
                         hipMemcpyDefault));

    // Free device memory
    HIP_ERRCHK(hipFree(d_x));
    HIP_ERRCHK(hipFree(d_y));

    printf("reference: %f %f %f %f ... %f %f\n", x[0], x[1], x[2],
           x[3], x[num_values - 2], x[num_values - 1]);
    printf("   result: %f %f %f %f ... %f %f\n", y[0], y[1], y[2], y[3],
           y[num_values - 2], y[num_values - 1]);

    // Check result of computation on the GPU
    double error = 0.0;
    for (size_t i = 0; i < num_values; i++) {
        error += abs(x[i] - y[i]);
    }

    printf("total error: %f\n", error);
    printf("  reference: %f at (42,42)\n", x[42 * num_rows + 42]);
    printf("     result: %f at (42,42)\n", y[42 * num_rows + 42]);

    index_visualisation::output();

    return 0;
}

namespace index_visualisation {

/*
 * Bonus: if you're interested on the index calculation of a 2D kernel,
 * you can use the function here to output a bunch of values to files and
 * inspect them.
 * */

__global__ void output_indices(int *tx, int *ty, int *bx, int *by, int *cols,
                               int *rows, int *col_starts, int *row_starts,
                               int *indices, int *block_idx, size_t num_cols,
                               size_t num_rows) {
    const size_t col_start = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t row_start = threadIdx.y + blockIdx.y * blockDim.y;
    const size_t col_stride = blockDim.x * gridDim.x;
    const size_t row_stride = blockDim.y * gridDim.y;

    for (size_t row = row_start; row < num_rows; row += row_stride) {
        for (size_t col = col_start; col < num_cols; col += col_stride) {
            const size_t index = row * num_cols + col;
            tx[index] = threadIdx.x;
            ty[index] = threadIdx.y;
            bx[index] = blockIdx.x;
            by[index] = blockIdx.y;
            cols[index] = col;
            rows[index] = row;
            col_starts[index] = col_start;
            row_starts[index] = row_start;
            indices[index] = index;
            block_idx[index] = blockIdx.x + blockIdx.y * gridDim.x;
        }
    }
}

void output() {
    // Changing the number of columns and rows
    // and the threads/blocks configuration below
    // produces different kind of output.
    static constexpr size_t num_cols = 40;
    static constexpr size_t num_rows = 80;
    static constexpr size_t num_values = num_cols * num_rows;
    static constexpr size_t num_bytes = sizeof(int) * num_values;

    void *tx = nullptr;
    void *ty = nullptr;
    void *bx = nullptr;
    void *by = nullptr;
    void *cols = nullptr;
    void *rows = nullptr;
    void *col_starts = nullptr;
    void *row_starts = nullptr;
    void *indices = nullptr;
    void *block_idx = nullptr;

    HIP_ERRCHK(hipMalloc(&tx, num_bytes));
    HIP_ERRCHK(hipMalloc(&ty, num_bytes));
    HIP_ERRCHK(hipMalloc(&bx, num_bytes));
    HIP_ERRCHK(hipMalloc(&by, num_bytes));
    HIP_ERRCHK(hipMalloc(&cols, num_bytes));
    HIP_ERRCHK(hipMalloc(&rows, num_bytes));
    HIP_ERRCHK(hipMalloc(&col_starts, num_bytes));
    HIP_ERRCHK(hipMalloc(&row_starts, num_bytes));
    HIP_ERRCHK(hipMalloc(&indices, num_bytes));
    HIP_ERRCHK(hipMalloc(&block_idx, num_bytes));

    const dim3 threads(16, 16, 1);
    const dim3 blocks(2, 2, 1);

    // clang-format off
    output_indices<<<blocks, threads>>>(
        static_cast<int *>(tx),
        static_cast<int *>(ty),
        static_cast<int *>(bx),
        static_cast<int *>(by),
        static_cast<int *>(cols),
        static_cast<int *>(rows),
        static_cast<int *>(col_starts),
        static_cast<int *>(row_starts),
        static_cast<int *>(indices),
        static_cast<int *>(block_idx),
        num_cols,
        num_rows
        );
    // clang-format on

    // Copy results back to CPU
    std::vector<int> htx(num_values);
    std::vector<int> hty(num_values);
    std::vector<int> hbx(num_values);
    std::vector<int> hby(num_values);
    std::vector<int> hcols(num_values);
    std::vector<int> hrows(num_values);
    std::vector<int> hcol_starts(num_values);
    std::vector<int> hrow_starts(num_values);
    std::vector<int> hindices(num_values);
    std::vector<int> hblock_idx(num_values);
    HIP_ERRCHK(hipMemcpy(static_cast<void *>(htx.data()), tx, num_bytes,
                         hipMemcpyDefault));
    HIP_ERRCHK(hipMemcpy(static_cast<void *>(hty.data()), ty, num_bytes,
                         hipMemcpyDefault));
    HIP_ERRCHK(hipMemcpy(static_cast<void *>(hbx.data()), bx, num_bytes,
                         hipMemcpyDefault));
    HIP_ERRCHK(hipMemcpy(static_cast<void *>(hby.data()), by, num_bytes,
                         hipMemcpyDefault));
    HIP_ERRCHK(hipMemcpy(static_cast<void *>(hrows.data()), rows, num_bytes,
                         hipMemcpyDefault));
    HIP_ERRCHK(hipMemcpy(static_cast<void *>(hcols.data()), cols, num_bytes,
                         hipMemcpyDefault));
    HIP_ERRCHK(hipMemcpy(static_cast<void *>(hrow_starts.data()), row_starts,
                         num_bytes, hipMemcpyDefault));
    HIP_ERRCHK(hipMemcpy(static_cast<void *>(hcol_starts.data()), col_starts,
                         num_bytes, hipMemcpyDefault));
    HIP_ERRCHK(hipMemcpy(static_cast<void *>(hindices.data()), indices,
                         num_bytes, hipMemcpyDefault));
    HIP_ERRCHK(hipMemcpy(static_cast<void *>(hblock_idx.data()), block_idx,
                         num_bytes, hipMemcpyDefault));

    // Free device memory
    HIP_ERRCHK(hipFree(tx));
    HIP_ERRCHK(hipFree(ty));
    HIP_ERRCHK(hipFree(bx));
    HIP_ERRCHK(hipFree(by));
    HIP_ERRCHK(hipFree(rows));
    HIP_ERRCHK(hipFree(cols));
    HIP_ERRCHK(hipFree(row_starts));
    HIP_ERRCHK(hipFree(col_starts));
    HIP_ERRCHK(hipFree(indices));
    HIP_ERRCHK(hipFree(block_idx));

    // Output values
    auto write = [](const char *name, std::vector<int> &buffer) {
        std::stringstream ss;
        for (size_t row = 0; row < num_rows; row++) {
            for (size_t col = 0; col < num_cols; col++) {
                const size_t index = row * num_cols + col;
                ss << buffer[index];

                if (col == num_cols - 1) {
                    break;
                }

                ss << " ";
            }

            if (row == num_rows - 1) {
                break;
            }

            ss << "\n";
        }

        std::ofstream file(name);
        if (file.is_open()) {
            file << ss.str();
        }
    };

    write("tx.txt", htx);
    write("ty.txt", hty);
    write("bx.txt", hbx);
    write("by.txt", hby);
    write("rows.txt", hrows);
    write("cols.txt", hcols);
    write("row_starts.txt", hrow_starts);
    write("col_starts.txt", hcol_starts);
    write("indices.txt", hindices);
    write("block_idx.txt", hblock_idx);
}
} // namespace index_visualisation
