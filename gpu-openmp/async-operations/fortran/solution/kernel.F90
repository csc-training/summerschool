module mandelbrot_kernel
  use constants
  implicit none

contains
  integer function kernel(xi, yi)
    !$omp declare target
    integer, intent(in) :: xi, yi

    real(dp) :: x0, y0, x, y, z
    integer :: i

    x0 = xmin + xi * dx
    y0 = ymin + yi * dy
    x = 0.0
    y = 0.0

    i = 0
    do while ((i < max_iters) .and. (x**2 + y**2 < 4.0))
      z = x**2 - y**2 + x0
      y = 2 * x * y + y0
      x = z
      i = i + 1
    end do

    kernel = i
  end function

end module

