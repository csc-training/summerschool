program main
  use iso_fortran_env, only : REAL64
  use io

  implicit none

  integer, parameter :: dp = REAL64
  real(dp), dimension(:,:), allocatable :: field



  call read_field(field, 'bottle.dat')

  call write_field(field, 0)

end program main
