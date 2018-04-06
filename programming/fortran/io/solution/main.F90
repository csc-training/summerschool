program main
  use io
  implicit none
  real, dimension(:,:), allocatable :: field

  call read_field(field, 'bottle.dat')

  call write_field(field, 0)

end program main
