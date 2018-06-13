program intrinsics
  implicit none
  integer :: nx, ny
  integer :: i, j, alloc_stat
  real, dimension(:,:), allocatable :: A
  real :: dx, dy, x, y

  write (*,*) 'Give number of rows and columns for matrix A:'
  read (*,*) nx, ny
  dx = 1.0/(nx-1)
  dy = 1.0/(ny-1)

  allocate(A(nx,ny), stat = alloc_stat)
  if (alloc_stat /= 0) call abort()

  ! Initializing array
  y = 0.0
  do j = 1, ny
     x = 0.0
     do i = 1, nx
        A(i, j) = x**2 + y**2
        x = x + dx
     end do
     y = y + dy
  end do

  do i = 1, nx
     write(*,'(*(F6.1))') A(i,:)
  end do

  ! TODO:
  ! Using array intrinsics to get information from array A

  write (*,*) 'a) Sum of elements across 2nd dimension of A: ', sum(A,2)

  write (*,*) 'b) Coordinates of maximum element: ', maxloc(A)

  write (*,*) 'c) Absolute minimum value: ', minval(A)

  write (*,*) 'd) Are elements of A greater than or equal to 0: ', any(A >= 0.0)

  write (*,*) 'e) Number of elements greater than or equal to 0.5: ', count(A >= 0.5)

end program intrinsics
