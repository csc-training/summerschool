program jacobi
  use iso_fortran_env, only : REAL64
  use omp_lib
  implicit none

  integer, parameter :: dp = REAL64
  integer, parameter :: nx = 2000
  integer, parameter :: ny = 2000
  integer, parameter :: niter = 500
  real(dp), parameter :: factor = 0.25

  integer :: i, j, iter

  real(dp), dimension(nx, ny) :: u, unew

  real(dp) :: t0, t1

  ! Initialize u
  do j = 1, ny
    do i = 1, nx
      u(i, j) = (i - nx / 2)**2 / nx + (i - ny / 2)**2 / ny
    end do
  end do
                      
  t0 = omp_get_wtime()
  ! Iterate
  do iter = 1, niter
    ! Stencil update 1
    !$omp target teams distribute
    do j = 2, ny - 1
      !$omp parallel do
      do i = 2, nx - 1
        unew(i, j) = factor * (u(i + 1, j) - 2.0 * u(i, j) + u(i - 1, j) + &
                               u(i, j + 1) - 2.0 * u(i, j) + u(i, j - 1))
      end do
    end do

    ! "Swap" the arrays, stencil update 2
    !$omp target teams distribute
    do j = 2, ny - 1
      !$omp parallel do
      do i = 2, nx - 1
        u(i, j) = factor * (unew(i + 1, j) - 2.0 * unew(i, j) + unew(i - 1, j) + &
                            unew(i, j + 1) - 2.0 * unew(i, j) + unew(i, j - 1))
      end do
    end do
  end do

  t1 = omp_get_wtime();
  ! Check final result
  write(*,*) "u(1,1) = ", u(1, 1)
  write(*, *) "Time spent: ", t1 - t0

end program
