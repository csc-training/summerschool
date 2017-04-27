program loops
  implicit none
  integer, parameter :: n = 16, m = 16
  real, dimension(n, m) :: A
  integer :: i, j


! Initialize everything first to zero 0.0
! Remember that now we have A(ydir, xdir) 
! so we define height with n and width with m.

! For clarity we associate i with y and j with x
  do i = 2, n-1
    do j = 2, m-1
      A(i,j) = 0.0
    end do
  end do

! Top
  do j = 1, m
    A(1,j) = 1.0
  end do

! Bottom
  do j = 1, m
    A(n,j) = 2.0
  end do

! Left
  do i = 2, n-1
    A(i,1) = 3.0
  end do

! Right
  do i = 2, n-1
    A(i,m) = 4.0
  end do



!--------------------------------------------------
! printing of the final array
  do i = 1, n
     write(*,'(*(F4.1))') A(i,:)
  end do

end program loops
