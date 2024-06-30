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
    int nx, ny;

    #pragma omp parallel shared(norm, iter, nx, ny, eps, u, unew, b)  // Most of these variables are shared by default, but it is good programming practice to declare everything.
    {

    // TODO start: add necessary execution controls (single, master, barrier)
    //             in this parallel region
    #pragma omp single
    {
    // Read b
    read_file(b);

    nx = b.nx;
    ny = b.ny;

    // Allocate space also for  boundaries
    u.allocate(nx + 2, ny + 2);
    }

    // Initialize
    #pragma omp for
    for (int i=0; i < nx + 2; i++)
        for (int j=0; j < ny + 2; j++) {
            u(i, j) = 0.0;
        }

    #pragma omp single
    unew = u;

    // Jacobi iteration
    do {
        // Without barrier here, norm may be set to zero before some threads have 
        // finished the check at the "while". Remember that norm is shared.
        #pragma omp barrier
        {
        #pragma omp single  // Only one thread needs to reset norm.
        norm = 0.0;
        }

        #pragma omp for reduction(+:norm)
        for (int i=1; i < nx + 1; i++)  // i and j private? They are only defined in the scope of the for loop, so it is fine.
            for (int j=1; j < ny + 1; j++){
                // Every thread calculates their own unew(i, j), but all threads can read the entire unew, as it is shared.
                unew(i, j) = 0.25 * (u(i, j - 1) + u(i, j + 1) + 
                                    u(i - 1, j) + u(i + 1, j) -
                                    // b does not contain boundaries
                                    b(i - 1, j - 1));
                norm += (unew(i, j) - u(i, j)) * (unew(i, j) - u(i, j));
            } 

        //printf("Thread id: %d, iteration: %d, %f\n", omp_get_thread_num(), iter, norm);
        //#pragma omp barrier
        {
        // Swap unew and u after all threads have finished their part of the calculation, but 
        // only one swapping should occur, so that all other thread need to wait.
        #pragma omp single
        std::swap(unew, u);
        }

        #pragma omp master
        {
        if (iter % 500 == 0)
            std::cout << "Iteration " << iter << " norm: " << norm << std::endl;
        iter++;
        }  
    
        //#pragma omp barrier   // Setting the barrier here does not help, as there may be a very fast thread that finishes
                                // the while loop evaluation, reaches the norm reset, sets norm=0, before the other threads 
                                // have reached the while loop evaluation. Thus, the slower threds will exit the while loop 
                                // incorrectly. Having #pragma omp single for the norm reset does not stop this from happening,
                                // since the implicit barrier in #pragma omp single may just stop the slower threads before they 
                                // have evaluated the while loop condition.
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
