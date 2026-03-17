#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include "./error_checking.hpp"

const int WIDTH  = 16;
const int HEIGHT = 8;   // Intentionally different from WIDTH

__global__ void strideKernel(int* out) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    if (row < HEIGHT && col < WIDTH) {
        int idx = row * WIDTH + col;
        out[idx] = row * 100 + col;
        printf("Tid is(%d,%d) row %d col %d -> idx %2d value %3d\n",threadIdx.y,threadIdx.x,row, col, idx, out[idx]);
    }
}

int main() {
    const int N = WIDTH * HEIGHT;
    const int bytes = N * sizeof(int);

    int* d_out;
    HIP_ERRCHK(hipMalloc(&d_out, bytes));

    dim3 block(WIDTH, HEIGHT);

    std::vector<int> host(N);

    LAUNCH_KERNEL(strideKernel, 1, block, 0, 0, d_out);
    HIP_ERRCHK(hipDeviceSynchronize());

    HIP_ERRCHK(hipMemcpy(host.data(), d_out, bytes, hipMemcpyDeviceToHost));

    // Print compact tables to show the pattern difference
    auto dump = [&](const char* tag, const std::vector<int>& buf) {
        std::cout << "\n" << tag << ":\n";
        for (int r = 0; r < HEIGHT; r++) {
            for (int c = 0; c < WIDTH; c++) {
                int idx = r * WIDTH + c;  // always correct view when printing
                std::cout << std::setw(4) << buf[idx] << " ";
            }
            std::cout << "\n";
        }
    };

    dump("OUTPUT", host);

    HIP_ERRCHK(hipFree(d_out));

    return 0;
}
