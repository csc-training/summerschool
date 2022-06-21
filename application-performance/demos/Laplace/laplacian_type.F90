program laplacian
  use iso_fortran_env, only : REAL64
  use omp_lib
  implicit none

  integer, parameter :: dp = REAL64

  type :: field
     integer :: nx
     integer :: ny
     real(dp) :: dx
     real(dp) :: dy
     real(dp), dimension(:,:), allocatable :: data
  end type field


  type(field) :: A, L
  real(dp) :: x, y
  real(dp) :: meanL
  integer :: i, j
  integer :: iter
  integer, parameter :: niters = 50

  real(dp) :: t0, t1

  A%nx = 4096
  A%ny = 4096
  
  ! Grid spacing
  A%dx = 1.0/real(A%nx-1)
  A%dy = 1.0/real(A%ny-1)
  ! TODO: allocate matrices
  allocate (A%data(A%nx, A%ny), L%data(A%nx, A%ny))

  ! initialize array A(x,y) = (x^2 + y^2) in the domain [0:1,0:1]
  y = 0.0
  do j = 1, A%ny
     x = 0.0
     do i = 1, A%nx
        A%data(i,j) =  x**2 + y**2
        x = x + A%dx
     end do
     y = y + A%dy
  end do

  t0 = omp_get_wtime()
  ! Compute Laplacian of A and save it to L
  L%data = 0.0
  do iter = 1, niters
  do i = 2, A%nx-1
     do j = 2, A%ny-1
        L%data(i,j) = (A%data(i-1,j) - 2.0*A%data(i,j) + A%data(i+1,j)) / A%dx**2 + &
             (A%data(i,j-1) - 2.0*A%data(i,j) + A%data(i,j+1)) / A%dy**2
     end do
  end do
  end do
  t1 = omp_get_wtime()

  ! check the result
  meanL = 0.0
  do j = 2, A%ny-1
     do i = 2, A%nx-1
       meanL = meanL + L%data(i,j)
     end do
  end do

  meanL = meanL / ((A%nx - 1) * (A%ny - 1))


  write(*,*) 'numerical solution', meanL
  write(*,*) 'analytic solution', 4.0

  write(*,*) 'time', t1 - t0


end program laplacian
