program loops
  implicit none
! TODO define parameters n and m
! TODO: define floating point array A
  integer :: i, j


! TODO initialize array A here
! Remember that now we have A(ydir, xdir) 
! so we define height with n and width with m.




!--------------------------------------------------
! printing of the final array
  do i = 1, n
     write(*,'(*(F4.1))') A(i,:)
  end do

end program loops
