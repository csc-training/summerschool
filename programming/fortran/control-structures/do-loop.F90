program loops
  implicit none
  ! TODO define parameters nx and ny
  ! TODO: define real-valued array A
  integer :: i, j

  ! TODO initialize array A here



  !--------------------------------------------------
  ! Print out the array
  ! the ':' syntax means the whole row, see the Fortran arrays lecture
  do i = 1, nx
     write(*, '(12F6.1)') A(i,:)
  end do

end program loops
