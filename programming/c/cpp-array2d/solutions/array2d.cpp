
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
	    			"dynamic_array.png", 
            'c');

    if (error_code == 0) {
        printf("Wrote output file dynamic_array.png\n");
    } else {
        printf("Error while writing output file dynamic_array.png\n");
    }

}




int main() 
{

  Matrix<double> mat(10, 10);


  // Initial conditions for left and right
  for (int i = 0; i < mat.nx; i++) {
    mat(i, 0     )   = 20.0;
    mat(i, mat.ny-1) = 70.0;
  }
    
  // and top and bottom boundaries
  for (int j = 0; j < mat.ny; j++) {
    mat(0,        j) = 85.0;
    mat(mat.nx-1, j) = 5.0;
  }



  // print out result
  print_field(mat);

}
