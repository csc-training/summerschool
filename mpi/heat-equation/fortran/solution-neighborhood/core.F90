! Main solver routines for heat equation solver
module core
    use heat

contains

  ! Exchange the boundary data between MPI tasks
  subroutine exchange(field0, parallel)

    implicit none

    type(field), intent(inout) :: field0
    type(parallel_data), intent(in) :: parallel

    type(mpi_datatype) :: types(4)
    integer :: counts(4) = [1, 1, 1, 1]
    integer(kind=MPI_ADDRESS_KIND) :: sdisps(4), rdisps(4), disp0

    integer :: i, ierr

    types(1) = parallel%rowtype
    types(2) = parallel%rowtype
    types(3) = parallel%columntype
    types(4) = parallel%columntype

    ! calculate displacements with mpi_get_ddress    
    call mpi_get_address(field0%data(0, 0), disp0)

    call mpi_get_address(field0%data(1, 0), sdisps(1))
    call mpi_get_address(field0%data(field0%nx, 0), sdisps(2))
    call mpi_get_address(field0%data(0, 1), sdisps(3))
    call mpi_get_address(field0%data(0, field0%ny), sdisps(4))

    call mpi_get_address(field0%data(0, 0), rdisps(1))
    call mpi_get_address(field0%data(field0%nx + 1, 0), rdisps(2))
    call mpi_get_address(field0%data(0, 0), rdisps(3))
    call mpi_get_address(field0%data(0, field0%ny + 1), rdisps(4))

    do i = 1, 4
       sdisps(i) = sdisps(i) - disp0
       rdisps(i) = rdisps(i) - disp0
    end do
    
    call mpi_neighbor_alltoallw(field0%data, counts, sdisps, types, &
                                field0%data, counts, rdisps, types, &
                                parallel%comm, ierr)

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
