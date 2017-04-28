program arrays
  implicit none
  integer :: nx, ny
  integer :: j, alloc_stat
! TODO: define allocatable array A

  write (*,*) 'Give x and y dimensions of matrix A:'
  read (*,*) nx, ny

! TODO allocate A now that we know nx and ny

! TODO Use array syntax to initialize the array


!--------------------------------------------------
! Print out the arrays
  do j = 1, ny
     write(*,'(*(F5.2))') A(:,j)
  end do


end program arrays
