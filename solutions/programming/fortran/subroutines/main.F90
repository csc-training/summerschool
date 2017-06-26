program subroutines
  use laplacian_mod
  implicit none
  real, dimension(:,:), allocatable :: previous, current
  integer :: nx, ny

  write (*,*) 'Give x and y dimensions of a matrix:'
  read (*,*) nx, ny

  allocate(previous(nx,ny), current(nx,ny))
! initialize the array as in Ex. 2
  call initialize(previous)
  call write_field(previous)

  ! compute the Laplacian
  call laplacian(current, previous)
  ! print the result array
  call write_field(current)
 
end program subroutines

