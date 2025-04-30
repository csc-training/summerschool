program pi_test

  use iso_fortran_env, only : INT64

  implicit none

  integer(kind=INT64) :: nsamples
  character(len=85) :: arg
  real :: pi1, pi2

  if (command_argument_count() /= 1) then
    STOP 'Usage pi N where N is the number of samples'
  end if

  call get_command_argument(1, arg)
  read(arg, *) nsamples

  pi1 = cpu_pi(nsamples)
  write(*,*) 'Pi calculated with CPU', pi1
  pi2 = gpu_pi(nsamples)
  write(*,*) 'Pi calculated with GPU', pi2

contains

  real function cpu_pi(n)
    implicit none
    integer(kind=INT64) :: n
    integer :: i, inside

    real, dimension(n) :: x, y

    call random_number(x)
    call random_number(y)

    inside = 0
    do i = 1, n
      if (x(i)**2 + y(i)**2 < 1.0) then
        inside = inside + 1
      end if
    end do

    cpu_pi = 4.0 * real(inside) / real(n)

  end function cpu_pi

  real function gpu_pi(n)
    use, intrinsic :: iso_c_binding
    use omp_lib
    use curand
    implicit none
    integer(kind=INT64) :: n
    integer :: i, inside

    integer(kind=c_size_t) :: stat

    integer(kind=c_size_t) :: gen

    ! real, dimension(:), allocatable :: x, y
    real :: x(n), y(n)

    ! allocate(x(n), y(n))

    inside = 0

!$omp target enter data map(alloc:x, y)

    stat = curandCreateGenerator(gen, CURAND_RNG_PSEUDO_DEFAULT)

!$omp target data use_device_ptr(x, y)
    stat = curandGenerateUniform(gen, x, n)
    stat = curandGenerateUniform(gen, y, n)
!$omp end target data

!$omp target loop reduction(+:inside)
    do i = 1, n
      if (x(i)**2 + y(i)**2 < 1.0) then
        inside = inside + 1
      end if
    end do
!$omp end target loop


!$omp target exit data map(delete:x, y)

    gpu_pi = 4.0 * real(inside) / real(n)

    ! deallocate(x, y)
  end function gpu_pi

end program pi_test
