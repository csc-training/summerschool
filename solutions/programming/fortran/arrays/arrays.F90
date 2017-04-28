program arrays
  implicit none
  integer :: nx, ny
  integer :: j, alloc_stat
  real, dimension(:,:), allocatable :: A

  write (*,*) 'Give x and y dimensions of matrix A:'
  read (*,*) nx, ny

  allocate(A(ny,nx), stat = alloc_stat)
  if (alloc_stat /= 0) call abort()

  ! initialize middle to zero
  A(:,:)  = 0.0

  ! first left and right
  A(1,:)  = 3.0
  A(nx,:) = 4.0

  ! then top and bottom to get corners correct
  A(:,1)  = 1.0
  A(:,ny) = 2.0


!--------------------------------------------------
! Print out the arrays
  do j = 1, ny
     write(*,'(*(F5.2))') A(:,j)
  end do


end program arrays
