module constants
  use iso_fortran_env, only : REAL64
  implicit none

  integer, parameter :: dp = REAL64
  integer, parameter :: width = 8192;
  integer, parameter :: height= 8192;
  integer, parameter :: max_iters = 100
  real(dp), parameter :: xmin = -1.7
  real(dp), parameter :: xmax = .5
  real(dp), parameter :: ymin = -1.2
  real(dp), parameter :: ymax = 1.2
  real(dp), parameter :: dx = (xmax - xmin) / width
  real(dp), parameter :: dy = (ymax - ymin) / height

end module
