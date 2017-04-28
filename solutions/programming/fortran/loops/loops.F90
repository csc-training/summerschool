program loops
  implicit none
  integer, parameter :: nx = 16, ny = 16
  real, dimension(ny, nx) :: A
  integer :: i, j


! Initialize everything first to zero 0.0
! Remember that now we have A(ydir, xdir) 
! so we define height with ny and width with nx.

! For clarity we associate i with x and j with y
  do i = 2, nx-1
    do j = 2, ny-1
      A(j,i) = 0.0
    end do
  end do

! Top
  do i = 1, nx
    A(1,i) = 1.0
  end do

! Bottom
  do i = 1, nx
    A(ny,i) = 2.0
  end do

! Left
  do j = 2, ny-1
    A(j,1) = 3.0
  end do

! Right
  do j = 2, ny-1
    A(j,nx) = 4.0
  end do



!--------------------------------------------------
! printing of the final array
  do j = 1, ny
     write(*,'(*(F4.1))') A(j,:)
  end do

end program loops
