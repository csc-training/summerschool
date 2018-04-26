
extern "C" {
#include "pngwriter.h"
}

#include "matrix.h"


// This function saves the field values to a png file
void print_field( Matrix<double>& f ) {
    int error_code;

    error_code = save_png( 
            (double*) &f(0,0),
            f.nx, 
            f.ny, 
	    			"array2d.png", 
            'c');

    if (error_code == 0) {
        printf("Wrote output file array2d.png\n");
    } else {
        printf("Error while writing output file array2d.png\n");
    }

}


// initialize the matrix boundaries
void initialize( Matrix<double>& f ) {

  // Initial conditions for left and right
  for (int i = 0; i < f.nx; i++) {
    f(i, 0     ) = 20.0;
    f(i, f.ny-1) = 70.0;
  }
    
  // and top and bottom boundaries
  for (int j = 0; j < f.ny; j++) {
    f(0,      j) = 85.0;
    f(f.nx-1, j) = 5.0;
  }

}


// Finite difference Laplacian
void laplacian( Matrix<double>& f, double dx, double dy) {

  Matrix<double> lapl(f.nx, f.ny);

  for (int i = 1; i < f.nx - 1; i++) {
    for (int j = 1; j < f.ny - 1; j++) {
      lapl(i,j) =
        ( f(i-1,j  ) - 2.0*f(i,j) + f(i+1,j  ) )/(dx*dx) +
        ( f(i,  j-1) - 2.0*f(i,j) + f(i  ,j+1) )/(dy*dy);
    }
  }

  f = lapl;
}



// main function
int main() 
{

  // create empty 2d matrix
  Matrix<double> mat(256, 256);

  // initialize 
  initialize(mat);


  // print out result of the Laplacian
  double dx = 0.01;
  double dy = 0.01;
  laplacian(mat, dx, dy);


  print_field(mat);

}
