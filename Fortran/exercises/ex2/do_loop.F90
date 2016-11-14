program circle
  implicit none
  integer, parameter :: nx = 16, ny = 16, dist = 16
! TODO: declare the array "field"
  real :: d
  integer :: i, j, ds2

! TODO: write the double loop as instructed

  do i = 0, ny+1
     write(*,'(*(F4.1))') field(i,:)
  end do
end program circle


