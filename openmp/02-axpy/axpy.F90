! SPDX-FileCopyrightText: 2010 CSC - IT Center for Science Ltd. <www.csc.fi>
!
! SPDX-License-Identifier: MIT

#include "helper_functions.F90"

! Array size
#ifndef N
#define N 102400
#endif

program axpy
  use helper_functions
  implicit none
  real(8) :: alpha
  real(8), dimension(N) :: x, y
  integer :: i
  real(8) :: frac

  print '(A, I0)', "Array size N = ", N

  ! Initialization
  alpha = 3.0d0
  do i = 1, N
    frac = 1.0d0 / real(N - 1, kind=8)
    x(i) = real(i - 1, kind=8) * frac
    y(i) = real(i - 1, kind=8) * frac * 100.0d0
  end do

  ! Print input values
  print '(A)', "Input:"
  print '(A, F8.4)', "a = ", alpha
  call print_array("x", x)
  call print_array("y", y)

  ! Calculate axpy
  ! TODO: Add OpenMP directives for parallelization
  do i = 1, N
    y(i) = y(i) + alpha * x(i)
  end do

  ! Print output values
  print '(A)', "Output:"
  call print_array("y", y)

end program axpy

