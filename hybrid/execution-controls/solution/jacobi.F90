program jacobi
  use iso_fortran_env, only : REAL64

  implicit none

  integer, parameter :: dp = REAL64

  real(dp), parameter :: eps = 0.005
  real(dp), dimension(:,:), allocatable :: u, unew, b
  real(dp), allocatable, dimension(:,:) :: tmp_u

  real(dp) :: norm
  integer :: array_shape(2)
  integer :: nx, ny, i, j, iter
  

  real(kind=dp) :: t_start, t_end

  iter = 0
  t_start = wtime()

!$omp parallel shared(u, unew, b, norm, iter) private(i, j, nx, ny) 

  ! Read b
  !$omp single
  call read_file(b)
  array_shape = shape(b)
  !$omp end single

  nx = array_shape(1)
  ny = array_shape(2)

  ! Allocate space also for boundaries
  !$omp single
  allocate(u(0:nx + 1, 0:ny + 1), unew(0:nx + 1, 0:ny + 1))
  !$omp end single

  ! Initialize
!$omp workshare
  u = 0.0
  unew = 0.0
!$omp endworkshare

  ! Jacobi iteration
  do
! Without barrier here, norm may be set to zero before some threads have 
! finished the "if" check in the end of the loop
!$omp barrier
!$omp single
    norm = 0.0
!$omp end single

!$omp do reduction(+:norm)
    do j = 1, ny
      do i = 1, nx
        unew(i, j) = 0.25 * (u(i, j - 1) + u(i, j + 1) + &
                          &  u(i - 1, j) + u(i + 1, j) - &
                          &  b(i, j))
        norm = norm + (unew(i, j) - u(i, j))**2
      end do
    end do
!$omp end do

    ! Swap u and une
!$omp single
    call move_alloc(u, tmp_u)
    call move_alloc(unew, u)
    call move_alloc(tmp_u, unew)
!$omp end single

!$omp master
    if (mod(iter, 500) == 0) then
      write(*,'(A, I6, A, F9.5)') 'Iteration', iter, ' norm:', norm
    end if
    iter = iter + 1
!$omp end master


    if (norm < eps) exit
  end do
!$omp end parallel

  t_end = wtime()
  write(*,'(A, I6, A, F9.5, A, F9.5)')  & 
       &     'Converged in', iter, ' iterations, norm', norm, &
       &     ' Time spent', t_end - t_start

  deallocate(b, u, unew)

contains

  subroutine read_file(mat)
    implicit none

    real(dp), allocatable, intent(inout) :: mat(:,:)

    integer :: nx, ny, i
    character(len=2) :: dummy


    open(10, file='input.dat')
    ! Read the header
    read(10, *) dummy, nx, ny

    allocate(mat(nx, ny))

       ! Read the data
    do i = 1, nx
       read(10, *) mat(i, 1:ny)
    end do 
    

  end subroutine read_file

  function wtime() result(t0)
#ifdef _OPENMP
    use omp_lib
#endif

    implicit none
    real(dp) :: t0

#ifdef _OPENMP
    t0 = omp_get_wtime()
#else
    call cpu_time(t0)
#endif

  end function wtime

end program jacobi
