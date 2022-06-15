program serial_pi
  use iso_fortran_env, only : REAL64
  implicit none

  integer, parameter :: dp = REAL64

  integer, parameter :: n = 840

  integer :: i, istart, istop
  real(dp) :: pi, x

  
  write(*,*) "Computing approximation to pi with n=", n

  istart = 1
  istop = n
     
  pi = 0.0
  do i = istart, istop
    x = (i - 0.5) / n
    pi = pi + 1.0 / (1.0 + x**2)
  end do
     
  pi = pi * 4.0 / n
  write(*,'(A,F18.16,A,F10.8,A)') 'Approximate pi=', pi, ' (exact pi=', 4.0*atan(1.0_dp), ')'
     
end program

