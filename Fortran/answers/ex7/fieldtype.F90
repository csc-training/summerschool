module laplacian_mod
  use iso_fortran_env, only : REAL64
  implicit none
  integer, parameter :: dp = REAL64

  type :: field
     integer :: nx
     integer :: ny
     real(dp) :: dx = 0.01
     real(dp) :: dy = 0.01
     real(dp), dimension(:,:), allocatable :: data
  end type field

end module laplacian_mod
