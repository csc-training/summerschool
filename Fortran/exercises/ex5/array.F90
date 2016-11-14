program array
  implicit none
  integer :: nx, ny
  integer :: i, alloc_stat
  ! TODO: define allocatable matrix A

  write (*,*) 'Give x and y dimensions of matrix A:'
  read (*,*) nx, ny

  ! TODO: Allocate A

  ! TODO: Fill A with random numbers

  do i = 1, ny
     write(*,'(*(F5.2))') A(i,:)
  end do

  ! TODO: Fill in the missing parts
  write (*,*) 'Sum of elements across 2nd dimension of A: '

  write (*,*) 'Coordinates of maximum element: ' 

  write (*,*) 'Absolute minimum value: ' 

  write (*,*) 'Are elements of A greater than or equal to 0: '

  write (*,*) 'Number of elements greater than or equal to 0.5: '


end program array
