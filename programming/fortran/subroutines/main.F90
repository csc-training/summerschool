program subroutines
  use laplacian_mod
  implicit none
  ! TODO: define the arrays
  integer :: nx, ny

  write (*,*) 'Give number of rows and columns for matrix A:'
  read (*,*) nx, ny

  allocate(previous(nx,ny), current(nx,ny))

  ! initialize the array
  call initialize(previous)

  call write_field(previous)

  ! compute the Laplacian
  call laplacian(current, previous)

  ! print the result array
  call write_field(current)

end program subroutines
