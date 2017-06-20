! Field metadata for heat equation solver
module heat
  use iso_fortran_env, only : REAL64
  implicit none

  integer, parameter :: dp = REAL64
  real(dp), parameter :: DX = 0.01, DY = 0.01  ! Fixed grid spacing

  type :: field
     integer :: nx          ! local dimension of the field
     integer :: ny
     integer :: nx_full     ! global dimension of the field
     integer :: ny_full
     real(dp) :: dx
     real(dp) :: dy
     real(dp), dimension(:,:), allocatable :: data
  end type field

  type :: parallel_data
     integer :: size
     integer :: rank
     integer :: nup, ndown, nleft, nright  ! Ranks of neighbouring MPI tasks
     integer :: comm
     integer :: rowtype                    ! MPI Datatype for communication of rows
     integer :: columntype                 ! MPI Datatype for communication of columns
     integer :: subarraytype               ! MPI Datatype for communication of inner region 

  end type parallel_data

contains
  ! Initialize the field type metadata
  ! Arguments:
  !   field0 (type(field)): input field
  !   nx, ny, dx, dy: field dimensions and spatial step size
  subroutine set_field_dimensions(field0, nx, ny, parallel)
    implicit none

    type(field), intent(out) :: field0
    integer, intent(in) :: nx, ny
    type(parallel_data), intent(in) :: parallel

    integer :: nx_local, ny_local

    nx_local = nx / sqrt(real(parallel%size))
    ny_local = ny / sqrt(real(parallel%size))

    field0%dx = DX
    field0%dy = DY
    field0%nx = nx_local
    field0%ny = ny_local
    field0%nx_full = nx
    field0%ny_full = ny

  end subroutine set_field_dimensions

  subroutine parallel_setup(parallel, nx, ny)
    use mpi
    
    implicit none
    
    type(parallel_data), intent(out) :: parallel
    integer, intent(in), optional :: nx, ny

    integer :: nx_local, ny_local
    integer :: world_size
    integer :: dims(2)
    logical :: periods(2) = (/.False., .FALSE./)
    integer, dimension(2) :: sizes, subsizes, offsets

    integer :: ierr

    call mpi_comm_size(MPI_COMM_WORLD, world_size, ierr)

    ! Set grid dimensions
    dims(1) = sqrt(real(world_size))
    dims(2) = dims(1)
    nx_local = nx / dims(1)
    ny_local = ny / dims(2)

    ! Ensure that the grid is divisible to the MPI tasks
    if (dims(1) * dims(2) /= world_size) then
       write(*,*) 'Cannot make square MPI grid, please use number of CPUs which is power of two', &
                   dims(1), dims(2), world_size
       call mpi_abort(MPI_COMM_WORLD, -1, ierr)
    end if
    if (nx_local * dims(1) /= nx) then
       write(*,*) 'Cannot divide grid evenly to processors in x-direction', &
                   nx_local, dims(1), nx
       call mpi_abort(MPI_COMM_WORLD, -2, ierr)
    end if
    if (ny_local * dims(2) /= ny) then
       write(*,*) 'Cannot divide grid evenly to processors in y-direction', &
                   ny_local, dims(2), ny
       call mpi_abort(MPI_COMM_WORLD, -2, ierr)
    end if

    ! Create cartesian communicator
    call mpi_cart_create(MPI_COMM_WORLD, 2, dims, periods, .true., parallel%comm, ierr)
    call mpi_cart_shift(parallel%comm, 0, 1, parallel%nup, parallel%ndown, ierr)
    call mpi_cart_shift(parallel%comm, 1, 1, parallel%nleft, &
                        parallel%nright, ierr)

    call mpi_comm_size(parallel%comm, parallel%size, ierr)
    call mpi_comm_rank(parallel%comm, parallel%rank, ierr)

    ! Create datatypes for halo exchange
    call mpi_type_vector(ny_local + 2, 1, nx_local + 2, MPI_DOUBLE_PRECISION, &
                         parallel%rowtype, ierr)
    call mpi_type_contiguous(nx_local + 2, MPI_DOUBLE_PRECISION, parallel%columntype, ierr)
    call mpi_type_commit(parallel%rowtype, ierr)
    call mpi_type_commit(parallel%columntype, ierr)

    ! Create datatype for subblock needed in I/O
    !   Rank 0 uses datatype for receiving data into full array while
    !   other ranks use datatype for sending the inner part of array
    subsizes(1) = nx_local
    subsizes(2) = ny_local
    offsets(1) = 0
    offsets(2) = 0
    if (parallel%rank == 0) then
       sizes(1) = nx
       sizes(2) = ny
    else
       sizes(1) = nx_local + 2
       sizes(2) = ny_local + 2
    end if
    call mpi_type_create_subarray(2, sizes, subsizes, offsets, MPI_ORDER_C, &
                             MPI_DOUBLE_PRECISION, parallel%subarraytype, ierr)
    call mpi_type_commit(parallel%subarraytype, ierr)
    
  end subroutine parallel_setup

end module heat
