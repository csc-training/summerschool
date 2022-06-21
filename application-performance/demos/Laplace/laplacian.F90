program laplacian
  use iso_fortran_env, only : REAL64
  use omp_lib
  implicit none

  integer, parameter :: dp = REAL64
  real(dp), dimension(:,:), allocatable :: A, L
  real(dp) :: dx, dy, x, y
  real(dp) :: meanL
  integer :: nx, ny, i, j
  integer :: iter
  integer, parameter :: niters=50

  real(dp) :: t0, t1

  nx = 4096
  ny = 4096

  ! Grid spacing
  dx = 1.0/real(nx-1)
  dy = 1.0/real(ny-1)
  ! TODO: allocate matrices
  allocate (A(nx, ny), L(nx, ny))

  ! initialize array A(x,y) = (x^2 + y^2) in the domain [0:1,0:1]
  y = 0.0
  do j = 1, ny
     x = 0.0
     do i = 1, nx
        A(i,j) =  x**2 + y**2
        x = x + dx
     end do
     y = y + dy
  end do

  t0 = omp_get_wtime()
  ! Compute Laplacian of A and save it to array L
  L = 0.0
  do iter = 1, niters
  do i = 2, nx-1
     do j = 2, ny-1
        L(i,j) = (A(i-1,j) - 2.0*A(i,j) + A(i+1,j)) / dx**2 + &
             (A(i,j-1) - 2.0*A(i,j) + A(i,j+1)) / dy**2
     end do
  end do
  end do
  t1 = omp_get_wtime()

  ! check the result
  meanL = 0.0
  do j = 2, ny-1
     do i = 2, nx-1
       meanL = meanL + L(i,j)
     end do
  end do

  meanL = meanL / ((nx - 1) * (ny - 1))


  write(*,*) 'numerical solution', meanL
  write(*,*) 'analytic solution', 4.0

  write(*,*) 'time', t1 - t0


end program laplacian
