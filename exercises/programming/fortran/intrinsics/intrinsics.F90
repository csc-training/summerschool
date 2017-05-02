program intrinsics
  implicit none
  integer :: nx, ny
  integer :: j, alloc_stat
  real, dimension(:,:), allocatable :: A

  write (*,*) 'Give x and y dimensions of matrix A:'
  read (*,*) nx, ny

  allocate(A(nx,ny), stat = alloc_stat)
  if (alloc_stat /= 0) call abort()

  ! Initializing array
  A(:,:)  = 65.0 ! middle
  A(:,1)  = 20.0 ! top
  A(:,ny) = 70.0 ! bottom
  A(1,:)  = 85.0 ! left
  A(nx,:) = 5.0  ! right

  !--------------------------------------------------
  ! Print out the arrays
  do j = 1, ny
     write(*,'(*(F5.1))') A(:,j)
  end do


  !--------------------------------------------------
  ! TODO: use array intrinsics to probe elements of A




end program intrinsics
