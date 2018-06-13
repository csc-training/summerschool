program laplacian
  use iso_fortran_env, only : REAL64
  implicit none

  integer, parameter :: dp = REAL64
  real(dp), dimension(:,:), allocatable :: A, L
  real(dp) :: dx, dy, x, y
  integer :: nx, ny, i, j

  write (*,*)  'Give number of rows and columns for matrix A:'
  read (*,*) nx, ny
  ! Grid spacing
  dx = 1.0/real(nx-1)
  dy = 1.0/real(ny-1)
  ! TODO: allocate matrices


  ! initialize array A(x,y) = (x^2 + y^2) in the domain [0:1,0:1]
  y = 0.0
  do j = 1, ny
     x = 0.0
     do i = 1, nx
        A(i,j) =  x**2 + y**2
        x = x + dx
     end do
     y = y + dy
  end do

  ! TODO: Compute Laplacian of A and save it to array L


  ! TODO: Printing of the arrays
  write(*,*) "Original array:"


  write(*,*) "Laplacian of the array:"


  ! Analytically, the Laplacian of the function is nabla^2 A(x,y) = 4


end program laplacian
