
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




// main function
int main() 
{

  // TODO create empty 2d matrix called mat

  // TODO implement initialization function


  double dx = 0.01;
  double dy = 0.01;

  // TODO implement function to compute Laplacian of mat


  print_field(mat);

}
