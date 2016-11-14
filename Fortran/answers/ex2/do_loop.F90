program circle
  implicit none
  integer, parameter :: nx = 16, ny = 16, dist = 16
  real, dimension(0:ny+1,0:nx+1) :: field
  real :: d
  integer :: i, j, ds2

  do j = 0, nx + 1
     do i = 0, ny + 1
        if ((i+j) < dist) then
           field(i,j) = 5.0
        else
           field(i,j) = 1.0
        end if
     end do
  end do

  do i = 0, ny + 1
     write(*,'(*(F4.1))') field(i,:)
  end do
end program circle


