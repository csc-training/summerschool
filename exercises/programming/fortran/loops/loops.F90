program loops
  implicit none
! TODO define parameters nx and ny
! TODO: define floating point array A
  integer :: i, j


! TODO initialize array A here
! Remember that we now have A(ydir, xdir),
! so we define height with ny and width with nx.




!--------------------------------------------------
! printing of the final array
  do j = 1, ny
     write(*,'(*(F4.1))') A(j,:)
  end do

end program loops
