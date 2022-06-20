/*
 * HAVE_DEF is set during compile time
 * and determines which accelerator backend is used
 * by including the respective header file
 */

#include <chrono>
#include <cmath>
#include <iostream>
#include <climits>
#include <random>
#include <stdio.h>

// Namespaces "comms" and "devices" declared here
#include "comms.h"

#define N_BESSEL 16
#define N_ITER 10000
#define N_POPU 10000
#define N_SAMPLE 50

int main(int argc, char *argv []){

  // Set timer
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  // Initialize processes and devices
  comms::init_procs(&argc, &argv);
  unsigned int my_rank = comms::get_rank();

  // Memory allocation
  float* b_error_mean = (float*)devices::allocate(N_BESSEL * sizeof(float));

  // Use a non-deterministic random number generator for the master seed value
  std::random_device rd;

  // Use 64 bit Mersenne Twister 19937 generator
  std::mt19937_64 mt(rd());

  // Get a random unsigned long long from a uniform integer distribution (srand requires 32b uint)
  std::uniform_int_distribution<unsigned long long> dist(0, UINT_MAX);

  // Get the non-deterministic random master seed value
  unsigned long long seed = dist(mt);

  // Initialize the mean error array
  devices::parallel_for(N_BESSEL, 
    DEVICE_LAMBDA(const int j) {
      b_error_mean[j] = 0.0f;
    }
  );

  // Run the loop over iterations
  devices::parallel_for(N_ITER, 
    DEVICE_LAMBDA(const int iter) {

      float p_mean = 0.0f;
      float s_mean = 0.0f;
      
      for(int i = 0; i < N_POPU; ++i){
        unsigned long long seq = ((unsigned long long)iter * (unsigned long long)N_POPU) + (unsigned long long)i;
        float rnd_val = devices::random_float(seed, seq, i, 100.0f, 15.0f);
        p_mean += rnd_val;
        if(i < N_SAMPLE) s_mean += rnd_val;
        if(iter == 0 && i < 3) printf("Rank %u, rnd_val[%d]: %.5f \n", my_rank, i, rnd_val);
      }
      
      p_mean /= N_POPU;
      s_mean /= N_SAMPLE;
      
      float b_stdev[N_BESSEL];
      float b_sum = 0.0f;
      float p_stdev = 0.0f;
      
      for(int i = 0; i < N_POPU; ++i){
        unsigned long long seq = ((unsigned long long)iter * (unsigned long long)N_POPU) + (unsigned long long)i;
        float rnd_val = devices::random_float(seed, seq, i, 100.0f, 15.0f);
        float p_diff = rnd_val - p_mean;
        p_stdev += p_diff * p_diff;
        if(i < N_SAMPLE){
          float b_diff = rnd_val - s_mean;
          b_sum += b_diff * b_diff;   
        }
        //if(iter == 0 && i < 3) printf("Rank %u, rnd_val[%d]: %.5f? \n", my_rank, i, rnd_val);
      }
      p_stdev /= N_POPU;
      p_stdev = sqrtf(p_stdev);
      //printf("p_stdev: %f\n",p_stdev);
      
      for(int j = 0; j < N_BESSEL; ++j){
        float sub = j * (1.6f / N_BESSEL);
        b_stdev[j] = b_sum / (N_SAMPLE - sub);
        b_stdev[j] = sqrtf(b_stdev[j]);
        float diff = p_stdev - b_stdev[j];
        //printf("b_stdev[%d]: %f, error[iter: %d][sub: %f]: %f\n", j, b_stdev[j], iter, sub, sqrt(diff * diff));  

        // Sum the errors of each iteration
        devices::atomic_add(&b_error_mean[j], diff * diff);
      }     
    }
  );

  // Each process sends its rank to reduction, root process collects the result
  comms::reduce_procs(b_error_mean, N_BESSEL);

  // Divide the error sum to find the averaged error for each tested Bessel value
  if(my_rank == 0){
    for(int j = 0; j < N_BESSEL; ++j){
      b_error_mean[j] /= (comms::get_procs() * N_ITER);
      float sub = j * (1.6f / N_BESSEL);
      printf("Mean squared error (MSE) for Bessel = %.2f is %.10f\n", sub, b_error_mean[j]);
    }
  }
  
  // Memory deallocations
  devices::free((void*)b_error_mean);

  // Finalize processes and devices
  comms::finalize_procs();

  // Print timing
  if(my_rank == 0){
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
  }

  return 0;
}