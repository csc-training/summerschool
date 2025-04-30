program rand_test
  use iso_c_binding
  use iso_fortran_env, only : INT64
  use hipfort
  use hipfort_check 
  use hipfort_hiprand

  implicit none

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
    implicit none
    integer(kind=INT64) :: n
    integer :: i, inside
    type(c_ptr) :: gen = c_null_ptr
    type(c_ptr) :: x_d,y_d
    real(c_float), allocatable,target :: x(:),y(:)
    integer(c_size_t) :: istat

    allocate(x(1:n))
    allocate(y(1:n))
    Nbytes=sizeof(x)

    inside = 0


    ! TODO start: allocate x and y in the device with OpenMP enter data

    ! TODO end


    istat= hiprandCreateGenerator(gen, HIPRAND_RNG_PSEUDO_DEFAULT)


    ! TODO start: use device pointer for HIP random generator calls
    istat= hiprandGenerateUniform(gen, c_loc(x), n)
    istat= hiprandGenerateUniform(gen, c_loc(y), n)
    ! TODO end

    ! TODO start: execute the loop in parallel in device
    do i = 1, n
      if (x(i)**2 + y(i)**2 < 1.0) then
        inside = inside + 1
      end if
    end do

    ! TODO end

    ! TODO start deallocate x and y in the device with OpenMP exit data

    ! TODO end
    
    gpu_pi = 4.0 * real(inside) / real(n)

   deallocate(x, y)
  end function gpu_pi
  end program
