program loops
  implicit none
  integer, parameter :: nx = 10, ny = 10
  real, dimension(nx,ny) :: A
  real :: dx, dy, x, y
  integer :: i, j

  dx = 1. / (nx - 1)
  dy = 1. / (ny - 1)
  y = 0.0
  do j = 1, ny
     x = 0.0
     do i = 1, nx
        A(i,j) = x**2 + y**2
        x = x + dx
     end do
     y = y + dy
  end do

  ! Print out the array
  ! the ':' syntax means the whole row, see the Fortran arrays lecture
  do i = 1, nx
     write(*, '(10F7.3)') A(i,:)
  end do

end program loops
