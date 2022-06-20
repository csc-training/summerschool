! Main solver routines for heat equation solver
module core
  use heat

contains

  ! Update the temperature values using five-point stencil
  ! Arguments:
  !   curr (type(field)): current temperature values
  !   prev (type(field)): temperature values from previous time step
  !   a (real(dp)): diffusivity
  !   dt (real(dp)): time step
  subroutine evolve(curr, prev, a, dt)

    implicit none

    type(field),target, intent(inout) :: curr, prev
    real(dp) :: a, dt
    integer :: i, j, nx, ny
    real(dp) :: dx, dy
    real(dp), pointer, contiguous, dimension(:,:) :: currdata, prevdata

    ! Help the compiler avoid being confused
    nx = curr%nx
    ny = curr%ny
    dx = curr%dx
    dy = curr%dy
    currdata => curr%data
    prevdata => prev%data

    ! Determine the temperature field at next time step As we have
    ! fixed boundary conditions, the outermost gridpoints are not
    ! updated.

!$omp target teams distribute parallel do map(tofrom:prevdata, currdata)
    do j = 1, ny
       do i = 1, nx
          currdata(i, j) = prevdata(i, j) + a * dt * &
               & ((prevdata(i-1, j) - 2.0 * prevdata(i, j) + &
               &   prevdata(i+1, j)) / dx**2 + &
               &  (prevdata(i, j-1) - 2.0 * prevdata(i, j) + &
               &   prevdata(i, j+1)) / dy**2)
       end do
    end do
!$omp end target teams distribute parallel do 
    
  end subroutine evolve

  ! Start a data region and copy temperature fields to the device
  !   curr (type(field)): current temperature values
  !   prev (type(field)): values from previous time step
  subroutine enter_data(curr, prev)
    implicit none
    type(field), target, intent(in) :: curr, prev
    real(kind=dp), pointer, contiguous :: currdata(:,:), prevdata(:,:)

    currdata => curr%data
    prevdata => prev%data

    !$omp target enter data map(to: currdata, prevdata)
  end subroutine enter_data

  ! End a data region and copy temperature fields back to the host
  !   curr (type(field)): current temperature values
  !   prev (type(field)): values from previous time step
  subroutine exit_data(curr, prev)
    implicit none
    type(field), target :: curr, prev
    real(kind=dp), pointer, contiguous :: currdata(:,:), prevdata(:,:)

    currdata => curr%data
    prevdata => prev%data

    !$omp target exit data map(from: currdata, prevdata)
  end subroutine exit_data

  ! Copy a temperature field from the device to the host
  !   temperature (type(field)): temperature field
  subroutine update_host(temperature)
    implicit none
    type(field), target :: temperature
    real(kind=dp), pointer, contiguous :: tempdata(:,:)

    tempdata => temperature%data

    !$omp target update from(tempdata)
  end subroutine update_host


end module core
