program laplacian
  implicit none

  integer, parameter :: nx = 10, ny = 10
  real, dimension(nx, ny) :: prev, lapl
  integer :: i, j

  real, parameter :: dx = 0.01, dy = 0.01



! initialize prev array with varying boundaries
  prev(:,:)  = 65.0 ! middle
  prev(:,1)  = 20.0 ! top
  prev(:,ny) = 70.0 ! bottom
  prev(1,:)  = 85.0 ! left
  prev(nx,:) = 5.0  ! right

! initialize lapl array to zeros
  lapl(:,:)  = 0.0 ! middle

!-------------------------------------------------- 


  ! TODO: implement Laplacian in double do-loop using prev 
  ! and saving to lapl array. Remember to evaluate it only
  ! at the inner points.






!--------------------------------------------------
! printing of the prev and lapl arrays
  write(*,*) "Previous array:"
  do j = 1, ny
     write(*,'(*(F5.1))') prev(:,j)
  end do

  write(*,*) "Laplacian of the array:"
  do j = 1, ny
     write(*,'(*(F5.1))') lapl(:,j)
  end do

end program laplacian
