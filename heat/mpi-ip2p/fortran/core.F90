! Main solver routines for heat equation solver
module core
  use heat

contains

  ! Exchange the boundary data between MPI tasks
  ! part 1: start communication
  subroutine exchange_init(field0, parallel)
    use mpi

    implicit none

    type(field), intent(inout) :: field0
    type(parallel_data), intent(inout) :: parallel
    integer :: ierr
    ! TODO
  end subroutine exchange_init

  ! Compute one time step of temperature evolution
  ! Arguments:
  !   curr (type(field)): current temperature values
  !   prev (type(field)): values from previous time step
  !   a (real(dp)): update equation constant
  !   dt (real(dp)): time step value
  ! Update only the border-independent part of the field
  subroutine evolve_interior(curr, prev, a, dt)
    implicit none
    type(field), intent(inout) :: curr, prev
    real(dp) :: a, dt
    integer :: i, j, nx, ny

    nx = curr%nx
    ny = curr%ny

    ! TODO
  end subroutine evolve_interior

  ! Finalize the non-blocking communication
  subroutine exchange_finalize(parallel)
    use mpi
    implicit none
    type(parallel_data), intent(inout) :: parallel
    integer :: ierr

    ! TODO
  end subroutine exchange_finalize

  ! Compute one time step of temperature evolution
  ! Arguments:
  !   curr (type(field)): current temperature values
  !   prev (type(field)): values from previous time step
  !   a (real(dp)): update equation constant
  !   dt (real(dp)): time step value
  ! Update only the border-dependent part
  subroutine evolve_edges(curr, prev, a, dt)

    implicit none

    type(field), intent(inout) :: curr, prev
    real(dp) :: a, dt
    integer :: i, j, nx, ny

    nx = curr%nx
    ny = curr%ny

    ! TODO

  end subroutine evolve_edges

end module core
