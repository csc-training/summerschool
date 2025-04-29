
program rand_test
  use iso_c_binding
  use iso_fortran_env, only : INT64

  implicit none

  type(c_ptr) :: x_d,y_d
  type(c_ptr) :: rand_gen
  integer(kind=INT64) :: nsamples
  character(len=85) :: arg
  real :: pi1, pi2
  integer(c_size_t):: Nbytes

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

    real, allocatable:: x(:),y(:)


    allocate(x(1:n))
    allocate(y(1:n))

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
  use hipfort
  use hipfort_check 
  use hipfort_hiprand
  use omp_lib
    implicit none
    integer(kind=INT64) :: n
    integer :: i, inside
    type(c_ptr) :: gen = c_null_ptr
    !integer(c_size_t) :: gen
    real(c_float), allocatable,target :: x(:),y(:)
    integer(c_size_t) :: istat

    allocate(x(1:n))
    allocate(y(1:n))
    inside = 0

!$omp target enter data map(alloc:x, y)

    istat= hiprandCreateGenerator(gen, HIPRAND_RNG_PSEUDO_DEFAULT)

!$omp target data use_device_ptr(x, y)
    istat= hiprandGenerateUniform(gen, c_loc(x), n)
    istat= hiprandGenerateUniform(gen, c_loc(y), n)
!$omp end target data

!$omp target teams distribute parallel do reduction(+:inside)
    do i = 1, n
      if (x(i)**2 + y(i)**2 < 1.0) then
        inside = inside + 1
      end if
    end do
!$omp end target teams distribute parallel do


!$omp target exit data map(delete:x, y)

    gpu_pi = 4.0 * real(inside) / real(n)

  deallocate(x,y)
  end function gpu_pi
  end program
