program subroutines
  use laplacian_mod
  implicit none
! TODO: define the arrays
  integer :: nx, ny, iter

  write (*,*) 'Give x and y dimensions of a matrix:'
  read (*,*) nx, ny

  allocate(previous(nx,ny), current(nx,ny))
  call initialize(previous)
  call write_field(previous)

  ! compute the Laplacian
  call laplacian(current, previous)
  ! print the result array
  call write_field(current)
 
end program subroutines

