program array
  implicit none
  integer :: nx, ny
  integer :: i, alloc_stat
  real, dimension(:,:), allocatable :: A

  write (*,*) 'Give x and y dimensions of matrix A:'
  read (*,*) nx, ny

  allocate(A(ny,nx), stat = alloc_stat)
  if (alloc_stat /= 0) call abort()

  call random_number(A)

  do i = 1, ny
     write(*,'(*(F5.2))') A(i,:)
  end do

  write (*,*) 'Sum of elements across 2nd dimension of A: ', sum(A,2) 

  write (*,*) 'Coordinates of maximum element: ', maxloc(A)

  write (*,*) 'Absolute minimum value: ', minval(A) 

  write (*,*) 'Are elements of A greater than or equal to 0: ', any(A >= 0.0)

  write (*,*) 'Number of elements greater than or equal to 0.5: ', count(A >= 0.5)


end program array
