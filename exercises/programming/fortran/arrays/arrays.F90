program arrays
  implicit none
  integer :: nx, ny
  integer :: i, alloc_stat
! TODO: define allocatable array A

  write (*,*) 'Give x and y dimensions of matrix A:'
  read (*,*) nx, ny

! TODO allocate A now that we know nx and ny

! TODO Use array syntax to initialize the array


!--------------------------------------------------
! Print out the arrays
  do i = 1, ny
     write(*,'(*(F5.2))') A(i,:)
  end do


end program arrays
