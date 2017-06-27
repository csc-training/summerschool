program array
  implicit none
  integer, parameter :: nx = 10, ny = 10
  real, dimension(0:nx+1,0:ny+1) :: A
  integer :: i

  ! Initialize the whole array in one go
  A = 65.0

  ! Adjust the boundaries 
  !--------------------------------------------------

  ! left column
  do i = 0, nx + 1
     A(i, 0) = 20.0
  end do
  
  ! right column
  do i = 0, nx + 1
     A(i, ny + 1) = 70.0
  end do
  
  ! top row
  do i = 0, ny + 1
     A(0, i) = 85.0
  end do
  
  ! bottom row
  do i = 0, ny+1
     A(nx + 1, i) = 5.0
  end do
  
  ! Print out the array
  ! the ':' syntax means the whole row, see the Fortran arrays lecture
  do i = 0, nx+1
     write(*, '(12F6.1)') A(i,:)
  end do

end program array
  
