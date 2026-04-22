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

    do j = 1, ny
       do i = 1, nx
          currdata(i, j) = prevdata(i, j) + a * dt * &
               & ((prevdata(i-1, j) - 2.0 * prevdata(i, j) + &
               &   prevdata(i+1, j)) / dx**2 + &
               &  (prevdata(i, j-1) - 2.0 * prevdata(i, j) + &
               &   prevdata(i, j+1)) / dy**2)
       end do
    end do
    
  end subroutine evolve

end module core
