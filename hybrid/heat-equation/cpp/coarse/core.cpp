// Main solver routines for heat equation solver

#include "heat.hpp"


// Update the temperature values using five-point stencil */
void evolve(Field& curr, const Field& prev, const double a, const double dt)
{

  // Compilers do not necessarily optimize division to multiplication, so make it explicit
  auto inv_dx2 = 1.0 / (prev.dx * prev.dx);
  auto inv_dy2 = 1.0 / (prev.dy * prev.dy);

  // Determine the temperature field at next time step
  // As we have fixed boundary conditions, the outermost gridpoints
  // are not updated.
  
  //#pragma omp parallel for collapse(2)  // Collapse nested loops into one loop for possible improved parallelisation. 
                                        // May give better performance if the outer loop is small compared to inner loop. 
                                        // For instance, if curr.nx=4, curr.ny=1000, being run on 16 threads, the 
                                        // curr.nx * curr.ny iterations will be divided amongst 16 threads. 
  
  #pragma omp parallel for              // Only the outer loop is parallelised. If the outer loop is smaller than the 
                                        // number of threads, there will be idle threads, and hence the collapse option 
                                        // may give better performance. E.g. if curr.nx=4, curr.ny=1000, being run on 16 
                                        // threads, curr.nx * curr.ny iterations will be divided amongst curr.nx=4 threads,
                                        // while the rest of the threads are idle.
                                        // This option may however be better as long as all threads are being kept occupied.
  for (int i = 1; i < curr.nx + 1; i++) { 
    for (int j = 1; j < curr.ny + 1; j++) { 
            curr(i, j) = prev(i, j) + a * dt * (
	       ( prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j) ) * inv_dx2 +
	       ( prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1) ) * inv_dy2
               );
    }
  }

}
