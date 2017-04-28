program loops
  implicit none
  integer, parameter :: nx = 16, ny = 16
  real, dimension(nx, ny) :: A
  integer :: i, j


! Initialize everything first to zero 0.0
! We define height with ny and width with nx.
! For clarity we associate i with x and j with y.


  do i = 2, nx-1
    do j = 2, ny-1
      A(i,j) = 0.0
    end do
  end do

! Top
  do i = 1, nx
    A(i,1) = 1.0
  end do

! Bottom
  do i = 1, nx
    A(i,ny) = 2.0
  end do

! Left
  do j = 2, ny-1
    A(1,j) = 3.0
  end do

! Right
  do j = 2, ny-1
    A(nx,j) = 4.0
  end do



!--------------------------------------------------
! printing of the final array
  do j = 1, ny
     write(*,'(*(F4.1))') A(:,j)
  end do

end program loops
