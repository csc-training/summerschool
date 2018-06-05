## Derived types

a) Define a derived type for a temperature field. Do the definition within a
module (let’s call it heat). The type contains the following elements: 

 - Number of grid points nx (=number of rows) and ny (=number of columns) (integers)
 - The grid spacings dx and dy in the x- and in the y-directions (real numbers)
 - An allocatable two-dimensional, real-valued array containing the data points 
  of the field. 

Define the real-valued variables into double precision, using the 
`ISO_FORTRAN_ENV` intrinsic module. 

b) Append this module with a subroutine that takes as input the number of
grid points in both dimensions and returns a field type where the 
metadata (grid points and grid spacings; let us set the latter to 0.01) 
has been initialized.



