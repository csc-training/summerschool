program arrays
  implicit none
  integer :: nx, ny
  integer :: i, alloc_stat
  ! TODO: define allocatable array A

  write (*,*) 'Give number of rows and columns for matrix A:'
  read (*,*) nx, ny

  ! TODO allocate A now that we know nx and ny


  ! TODO Use array syntax to initialize the array







  !--------------------------------------------------
  ! Print out the array
  do i = 1, nx
    write(*,'(*(F6.1))') A(i,:)
  end do


end program arrays
