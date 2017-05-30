! Main solver routines for heat equation solver
module core
    use heat

contains

  ! Exchange the boundary data between MPI tasks
  subroutine exchange(field0, parallel)
    use mpi

    implicit none

    type(field), intent(inout) :: field0
    type(parallel_data), intent(in) :: parallel

    integer :: ierr

    ! Send to left, receive from right
    call mpi_sendrecv(field0%data(0, 1), 1, parallel%columntype, &
                      parallel%nleft, 11, &
                      field0%data(0, field0%ny + 1), 1, parallel%columntype, &
                      parallel%nright, 11, &
                      parallel%comm, MPI_STATUS_IGNORE, ierr)

    ! Send to right, receive from left
    call mpi_sendrecv(field0%data(0, field0%ny), 1, parallel%columntype, &
                      parallel%nright, 12, &
                      field0%data(0, 0), 1, parallel%columntype, &
                      parallel%nleft, 12, &
                      parallel%comm, MPI_STATUS_IGNORE, ierr)

    ! Send to up receive from down
    call mpi_sendrecv(field0%data(1, 0), 1, parallel%rowtype, &
           parallel%nup, 13, field0%data(field0%nx+1, 0), 1, parallel%rowtype, &
           parallel%ndown, 13, parallel%comm, MPI_STATUS_IGNORE, ierr)

    ! Send to the down, receive from up
    call mpi_sendrecv(field0%data(field0%nx, 0), 1, parallel%rowtype, &
           parallel%ndown, 14, field0%data(0, 0), 1, parallel%rowtype, &
           parallel%nup, 14, parallel%comm, MPI_STATUS_IGNORE, ierr)

  end subroutine exchange

  ! Compute one time step of temperature evolution
  ! Arguments:
  !   curr (type(field)): current temperature values
  !   prev (type(field)): values from previous time step
  !   a (real(dp)): update equation constant
  !   dt (real(dp)): time step value
  subroutine evolve(curr, prev, a, dt)

    implicit none

    type(field), intent(inout) :: curr, prev
    real(dp) :: a, dt
    integer :: i, j, nx, ny

    nx = curr%nx
    ny = curr%ny

    do j = 1, ny
       do i = 1, nx
          curr%data(i, j) = prev%data(i, j) + a * dt * &
               & ((prev%data(i-1, j) - 2.0 * prev%data(i, j) + &
               &   prev%data(i+1, j)) / curr%dx**2 + &
               &  (prev%data(i, j-1) - 2.0 * prev%data(i, j) + &
               &   prev%data(i, j+1)) / curr%dy**2)
       end do
    end do
  end subroutine evolve

end module core
