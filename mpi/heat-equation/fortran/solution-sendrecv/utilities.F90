! Utility routines for heat equation solver

module utilities
  use heat

contains

  ! Swap the data fields of two variables of type field
  ! Arguments:
  !   curr, prev (type(field)): the two variables that are swapped
  subroutine swap_fields(curr, prev)

    implicit none

    type(field), intent(inout) :: curr, prev
    real(dp), allocatable, dimension(:,:) :: tmp

    call move_alloc(curr%data, tmp)
    call move_alloc(prev%data, curr%data)
    call move_alloc(tmp, prev%data)
  end subroutine swap_fields

  ! Copy the data from one field to another
  ! Arguments:
  !   from_field (type(field)): variable to copy from
  !   to_field (type(field)): variable to copy to
  subroutine copy_fields(from_field, to_field)

    implicit none

    type(field), intent(in) :: from_field
    type(field), intent(out) :: to_field

    ! Consistency checks
    if (.not.allocated(from_field%data)) then
       write (*,*) "Can not copy from a field without allocated data"
       stop
    end if
    if (.not.allocated(to_field%data)) then
       ! Target is not initialize, allocate memory
       allocate(to_field%data(lbound(from_field%data, 1):ubound(from_field%data, 1), &
            & lbound(from_field%data, 2):ubound(from_field%data, 2)))
    else if (any(shape(from_field%data) /= shape(to_field%data))) then
       write (*,*) "Wrong field data sizes in copy routine"
       print *, shape(from_field%data), shape(to_field%data)
       stop
    end if

    to_field%data = from_field%data

    to_field%nx = from_field%nx
    to_field%ny = from_field%ny
    to_field%nx_full = from_field%nx_full
    to_field%ny_full = from_field%ny_full
    to_field%dx = from_field%dx
    to_field%dy = from_field%dy
  end subroutine copy_fields

  function average(field0, parallelization)

    implicit none

    real(dp) :: average
    type(field) :: field0
    type(parallel_data), intent(in) :: parallelization

    real(dp) :: local_average
    integer :: i
    integer :: rc

    local_average = sum(field0%data(1:field0%nx, 1:field0%ny))
    ! Reduction is implemented here with point-to-point routines, any real code
    ! should use collective call
    if (parallelization%rank == 0) then
        average = local_average
        do i=1, parallelization%size-1
            call mpi_recv(local_average, 1, MPI_DOUBLE_PRECISION, i, 11, &
               &       MPI_COMM_WORLD, MPI_STATUS_IGNORE, rc)
            average = average + local_average
        end do
        average = average / (field0%nx_full * field0%ny_full)
    else
        call mpi_send(local_average, 1, MPI_DOUBLE_PRECISION, 0, 11, &
               &       MPI_COMM_WORLD, rc)
    end if

  end function average

end module utilities
