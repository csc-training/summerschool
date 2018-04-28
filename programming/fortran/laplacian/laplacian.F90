program laplacian
  use iso_fortran_env, only : REAL64
  implicit none

  integer, parameter :: nx = 10, ny = 10, dp = REAL64
  real(dp), dimension(nx,ny) :: U, lapl
  integer :: i, j

  real(dp) :: dx, dy, x, y

  ! Grid spacing
  dx = 1.0/real(nx)
  dy = 1.0/real(ny)

  ! TODO: 
  ! initialize array U(x,y) = (x^2 + y^2) in the domain [0:1,0:1] 










  ! TODO:
  ! Compute Laplacian of U and saving to lapl array.









  !--------------------------------------------------
  ! TODO:
  ! Printing of the prev and lapl arrays
  write(*,*) "Original array:"









 ! Analytically, the Laplacian of the function is nabla^2 U(x,y) = 4


end program laplacian
