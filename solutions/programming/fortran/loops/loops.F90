program array
  implicit none
  integer, parameter :: nx = 10, ny = 10
  real, dimension(0:nx+1,0:ny+1) :: field0
  integer :: i

  ! Initialize the whole array in one go
  field0 = 65.0

  ! Adjust the boundaries 
  do i = 0, nx + 1
     field0(i, 0) = 20.0
  end do
  
  do i = 0, nx + 1
     field0(i, ny + 1) = 70.0
  end do
  
  do i = 0, ny + 1
     field0(0, i) = 85.0
  end do
  
  do i = 0, ny+1
     field0(nx + 1, i) = 5.0
  end do
  
  ! Print out the array
  ! the ':' syntax means the whole row, see the Fortran arrays lecture
  do i = 0, nx+1
     write(*, '(12F6.1)') field0(i,:)
  end do

end program array
  
