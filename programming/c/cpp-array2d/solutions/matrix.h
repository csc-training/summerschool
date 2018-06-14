#include <vector>
#include <algorithm>
#include <assert.h>

// Generic 2D matrix array class.
//
// Internally data is stored in 1D vector but is
// accessed using index function that maps i and j
// indices to an element in the flat data vector.
//
// For easier usage, we overload parentheses () operator
// for accessing matrix elements in the usual (i,j)
// format.
//
template<typename T>
class Matrix
{

private:

    std::vector<T> data;

    // Internal 1D indexing
    const int indx(int i, int j) const {
        //assert that indices are reasonable
        assert(i >= 0 && i <  nx);
        assert(j >= 0 && j <  nx);

        return i + ny * j;
    }

public:

    // grid size along x and y
    int nx, ny;

    // Default constructor
    Matrix(int nx, int ny) : nx(nx), ny(ny) {
        data.resize(nx * ny);
    };

    // standard (i,j) syntax for setting elements
    T &operator()(int i, int j) {
        return data[ indx(i, j) ];
    }

    // standard (i,j) syntax for getting elements
    const T &operator()(int i, int j) const {
        return data[ indx(i, j) ];
    }


};
