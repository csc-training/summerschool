program arrays
  implicit none
  ! TODO: Define the array A
  real :: x, y, dx, dy
  integer :: nx, ny, i, j, alloc_stat

  write (*,*) 'Give number of rows and columns for matrix A:'
  read (*,*) nx, ny
  dx = 1.0/real(nx-1)
  dy = 1.0/real(ny-1)

  ! TODO: allocate the array A

  ! TODO: initalize the array A

  ! TODO: Print out the array


end program arrays
