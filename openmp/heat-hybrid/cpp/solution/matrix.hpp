#pragma once
#include <vector>
#include <cassert>

// Generic 2D matrix array class.
//
// Internally data is stored in 1D vector but is
// accessed using index function that maps i and j
// indices to an element in the flat data vector.
// Row major storage is used
// For easier usage, we overload parentheses () operator
// for accessing matrix elements in the usual (i,j)
// format.

template<typename T>
class Matrix
{

private:

    // Internal storage
    std::vector<T> _data;

    // Internal 1D indexing
    int indx(int i, int j) const {
        //assert that indices are reasonable
        assert(i >= 0 && i <  nx);
        assert(j >= 0 && j <  ny);

        return i * ny + j;
    }

public:

    // matrix dimensions
    int nx, ny;

    // Default constructor
    Matrix() = default;
    // Allocate at the time of construction
    Matrix(int nx, int ny) : nx(nx), ny(ny) {
        _data.resize(nx * ny);
    };

    void allocate(int nx_in, int ny_in) {
        nx = nx_in;
        ny = ny_in;
        _data.resize(nx * ny);
    };

    // standard (i,j) syntax for setting elements
    T& operator()(int i, int j) {
        return _data[ indx(i, j) ];
    }

    // standard (i,j) syntax for getting elements
    const T& operator()(int i, int j) const {
        return _data[ indx(i, j) ];
    }

    // provide possibility to get raw pointer for data at index (i,j) (needed for MPI)
    T* data(int i=0, int j=0) {return _data.data() + i * ny + j;}

};
