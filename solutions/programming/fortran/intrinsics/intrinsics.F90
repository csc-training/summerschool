program intrinsics
  implicit none
  integer :: nx, ny
  integer :: i, alloc_stat
  real, dimension(:,:), allocatable :: A

  write (*,*) 'Give number of rows and columns for matrix A:'
  read (*,*) nx, ny

  allocate(A(nx,ny), stat = alloc_stat)
  if (alloc_stat /= 0) call abort()

  ! Initializing array
  A(:,:)  = 65.0 ! middle
  A(:,1)  = 20.0 ! left
  A(:,ny) = 70.0 ! right
  A(1,:)  = 85.0 ! top
  A(nx,:) = 5.0  ! bottom

  !--------------------------------------------------
  ! Print out the array
  do i = 1, nx
    write(*,'(*(F6.1))') A(i,:)
  end do


  !--------------------------------------------------
  ! Using array intrinsics to get information from array A

  write (*,*) 'a) Sum of elements across 2nd dimension of A: ', sum(A,2) 

  write (*,*) 'b) Coordinates of maximum element: ', maxloc(A)

  write (*,*) 'c) Absolute minimum value: ', minval(A) 

  write (*,*) 'd) Are elements of A greater than or equal to 0: ', any(A >= 0.0)

  write (*,*) 'e) Number of elements greater than or equal to 0.5: ', count(A >= 0.5)

end program intrinsics
