#pragma once
#include "matrix.hpp"

// Forward declaration
struct ParallelData;

// Class for temperature field
struct Field {
    // nx and ny are the true dimensions of the field. The temperature matrix
    // contains also ghost layers, so it will have dimensions nx+2 x ny+2 
    int nx;                     // Local dimensions of the field
    int ny;
    int nz;
    int nx_full;                // Global dimensions of the field
    int ny_full;                // Global dimensions of the field
    int nz_full;                // Global dimensions of the field
    double dx = 0.01;           // Grid spacing
    double dy = 0.01;
    double dz = 0.01;

    Matrix<double> temperature;

#ifndef UNIFIED_MEMORY
    double *temperature_dev = NULL;
#endif

    void setup(int nx_in, int ny_in, int nz_in, ParallelData& parallel);

    void generate(const ParallelData& parallel);

    // standard (i,j) syntax for setting elements
    double& operator()(int i, int j, int k) {return temperature(i, j, k);}

    // standard (i,j) syntax for getting elements
    const double& operator()(int i, int j, int k) const {return temperature(i, j, k);}

    double* devdata(int i=0, int j=0, int k=0) {
#ifdef UNIFIED_MEMORY
       return temperature.data(i, j, k);
#else
       return temperature_dev + i * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
#endif
    }

};
