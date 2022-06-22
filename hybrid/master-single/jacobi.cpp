#ifdef _OPENMP
#include <omp.h>
#else
#include <chrono>
#endif

#include "matrix.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>

void read_file(Matrix<double>& mat);
double wtime();

// Solves Poisson equation with Jacobi iteration
// Source b is read from file "input.dat"
int main()
{
    constexpr double eps = 0.005;
    Matrix<double> u, unew, b;

    double norm;
    int iter = 0;

    double t_start = wtime();

    #pragma omp parallel shared(norm, iter)
    {

    // TODO start: add necessary execution controls (single, master, barrier)
    //             in this parallel region

    // Read b
    read_file(b);

    int nx = b.nx;
    int ny = b.ny;

    // Allocate space also for  boundaries
    u.allocate(nx + 2, ny + 2);

    // Initialize
    #pragma omp for
    for (int i=0; i < nx + 2; i++)
        for (int j=0; j < ny + 2; j++) 
            u(i, j) = 0.0;

    unew = u;

    // Jacobi iteration
    do {
        norm = 0.0;

        #pragma omp for reduction(+:norm)
        for (int i=1; i < nx + 1; i++)
            for (int j=1; j < ny + 1; j++) {
                unew(i, j) = 0.25 * (u(i, j - 1) + u(i, j + 1) + 
                                    u(i - 1, j) + u(i + 1, j) -
                                    // b does not contain boundaries
                                    b(i - 1, j - 1));
                norm += (unew(i, j) - u(i, j)) * (unew(i, j) - u(i, j));
            } 

        std::swap(unew, u);

        if (iter % 500 == 0)
            std::cout << "Iteration " << iter << " norm: " << norm << std::endl;
        iter++;    

    } while (norm > eps);

    // TODO end

    } // end parallel

    double t_end = wtime();

    std::cout << "Converged in " << iter << " iterations, norm " << norm 
              << " Time spent " << t_end - t_start << std::endl;

}

void read_file(Matrix<double>& mat)
{
    std::ifstream file;
    file.open("input.dat");
    if (!file) {
        std::cout << "Couldn't open the file 'input.dat'" << std::endl;
        exit(-1);
    }
    // Read the header
    std::string line, comment;
    std::getline(file, line);
    int nx, ny;
    std::stringstream(line) >> comment >> nx >> ny;

    mat.allocate(nx, ny);
    
    // Read the data
    
    for (int i = 0; i < nx; i++)
      for (int j = 0; j < ny; j++)
        file >> mat(i, j);

    file.close();
}

double wtime() {
#ifdef _OPENMP
  return omp_get_wtime();
#else
  using clock = std::chrono::high_resolution_clock;
  auto time = clock::now();
  auto duration = std::chrono::duration<double>(time.time_since_epoch());
  return duration.count();
#endif
}
