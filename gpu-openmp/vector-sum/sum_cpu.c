#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define NX 10240000

double wtime();

int main(void)
{
    double vecA[NX], vecB[NX], vecC[NX];

    /* Initialization of the vectors */
    #pragma omp target map(tofrom:vecA, vecB)  // Offload to GPU device
    #pragma omp teams  // Create a league of teams. teams and parallel constructs create threads, however, all the threads are still executing the same code
    #pragma omp distribute parallel for  // Distributes loop iterations over the teams.
    for (int i = 0; i < NX; i++) {
        vecA[i] = 1.0 / ((double) (NX - i));
        vecB[i] = vecA[i] * vecA[i];
    }

    // TODO: Implement vector addition vecC = vecA + vecB and use OpenMP
    //       for computing it in the device
    
    //Get the start time stamp 
    double start_clock = wtime();
    #pragma omp target map(to:vecA, vecB) map(tofrom:vecC)  // Offload to GPU device
    #pragma omp teams  // Create a league of teams. teams and parallel constructs create threads, however, all the threads are still executing the same code
    #pragma omp distribute parallel for  // Distributes loop iterations over the teams.
    //#pragma omp loop  //Leaves more freedom to the implementation to do the work division
    for (int i = 0; i < NX; i++) {
        vecC[i] = vecA[i] + vecB[i];
    }

    double stop_clock = wtime();
    printf("Vector sum calculation took %f seconds.\n", (stop_clock - start_clock));

    double sum = 0.0;
    /* Compute the check value */
    for (int i = 0; i < NX; i++) {
        sum += vecC[i];
    }
    printf("Reduction sum: %18.16f\n", sum);

    return 0;
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

