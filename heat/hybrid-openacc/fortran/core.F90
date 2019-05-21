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
    call mpi_sendrecv(field0%data(:, 1), field0%nx + 2, MPI_DOUBLE_PRECISION, &
         & parallel%nleft, 11, &
         & field0%data(:, field0%ny + 1), field0%nx + 2, MPI_DOUBLE_PRECISION, &
         & parallel%nright, 11, &
         & MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)

    ! Send to right, receive from left
    call mpi_sendrecv(field0%data(:, field0%ny), field0%nx + 2, MPI_DOUBLE_PRECISION, &
         & parallel%nright, 12, &
         & field0%data(:, 0), field0%nx + 2, MPI_DOUBLE_PRECISION,&
         & parallel%nleft, 12, &
         & MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)

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
    real(dp) :: dx, dy
    ! variables for memory access outside of a type
    real(dp), pointer, contiguous, dimension(:,:) :: currdata, prevdata

    ! HINT: to help the compiler do not access type components
    !       within OpenACC parallel regions
    nx = curr%nx
    ny = curr%ny
    dx = curr%dx
    dy = curr%dy
    currdata => curr%data
    prevdata => prev%data

    ! TODO: use OpenACC to parallelise the loops
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
