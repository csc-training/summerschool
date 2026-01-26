! SPDX-FileCopyrightText: 2025 CSC - IT Center for Science Ltd. <www.csc.fi>
!
! SPDX-License-Identifier: MIT

module helper_functions
  use, intrinsic :: iso_c_binding
  implicit none

#ifdef TRACE
  interface
#else
  contains
#endif

#ifdef TRACE
  function c_roctxRangePush(message) result(level) bind(C, name="roctxRangePushA")
    import :: c_int, c_char
#else
  function c_roctxRangePush(message) result(level)
#endif
    character(kind=c_char), intent(in) :: message(*)
    integer(c_int) :: level
#ifndef TRACE
    level = 0
#endif
  end function c_roctxRangePush

#ifdef TRACE
  function c_roctxRangePop() result(level) bind(C, name="roctxRangePop")
    import :: c_int
#else
  function c_roctxRangePop() result(level)
#endif
    integer(c_int) :: level
#ifndef TRACE
    level = 0
#endif
  end function c_roctxRangePop

#ifdef TRACE
  end interface
  contains
#endif

  subroutine print_array(name, x)
    character(len=*), intent(in) :: name
    real(8), intent(in) :: x(:)
    integer :: i, n
    integer, parameter :: PRINT_N = 4

    n = size(x)

    write(*,'(A)', advance='no') trim(name) // " ="
    do i = 1, n
      if (i <= PRINT_N .or. i > n - PRINT_N) then
        write(*,'(" ", F8.4)', advance='no') x(i)
      else if (i == PRINT_N + 1) then
        write(*,'(A)', advance='no') " ..."
      end if
    end do
    write(*,*)
  end subroutine print_array

  subroutine create_input(f)
    implicit none
    real(8), intent(out) :: f(:, :)
    integer :: i, j, ind, nx, ny
    real(8) :: cx, cy, sigma2, kx, ky, dx, dy, r2

    nx = size(f, 2)
    ny = size(f, 1)

    cx = real(nx, kind=8) / 2.0d0
    cy = real(ny, kind=8) / 2.0d0
    sigma2 = 0.05d0 * nx * ny
    kx = 20.0d0 / nx
    ky = 10.0d0 / ny

    do j = 1, nx
      do i = 1, ny
        dx = j - cx
        dy = i - cy
        r2 = dx * dx + dy * dy
        f(i,j) = cos(kx * dx + ky * dy) * exp(-r2 / sigma2)
      end do
    end do
  end subroutine create_input

  subroutine write_array(filename, array, ierr)
    implicit none
    character(len=*), intent(in) :: filename
    real(8), intent(in) :: array(:, :)
    integer, intent(out), optional :: ierr
    integer :: unit, ios, local_err
    integer(c_int) :: level

    level = c_roctxRangePush(c_char_"write_array")

    open(newunit=unit, file=filename, form='unformatted', access='stream', status='replace', action='write', iostat=ios)
    if (ios /= 0) then
      write(0,*) "Failed to open file"
      local_err = 1
      if (present(ierr)) ierr = local_err
      level = c_roctxRangePop()
      return
    end if

    write(unit) size(array, kind=8)
    write(unit, iostat=ios) array
    close(unit)

    if (ios /= 0) then
      write(0,*) "Failed to write all elements to file"
      local_err = 2
    else
      local_err = 0
    end if

    if (present(ierr)) ierr = local_err
    level = c_roctxRangePop()
  end subroutine write_array

end module helper_functions

