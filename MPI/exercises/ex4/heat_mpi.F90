
module heat_mpi
  use mpi
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
     integer :: nleft, nright  ! Ranks of neighbouring MPI tasks
  end type parallel_data

contains

  ! Initialize the field type metadata
  ! Arguments:
  !   field0 (type(field)): input field
  !   nx, ny, dx, dy: field dimensions and spatial step size
  subroutine initialize_field_metadata(field0, nx, ny, parallel)
    implicit none

    type(field), intent(out) :: field0
    integer, intent(in) :: nx, ny
    type(parallel_data), intent(in) :: parallel

    integer :: nx_local, ny_local

    nx_local = nx
    ny_local = ny / parallel%size

    field0%dx = DX
    field0%dy = DY
    field0%nx = nx_local
    field0%ny = ny_local
    field0%nx_full = nx
    field0%ny_full = ny

  end subroutine initialize_field_metadata

  ! Initialize parallelization data
  subroutine parallel_initialize(parallel, nx, ny)
    implicit none

    type(parallel_data), intent(out) :: parallel
    integer, intent(in), optional :: nx, ny

    integer :: ny_local
    integer :: ierr

    call mpi_comm_size(MPI_COMM_WORLD, parallel%size, ierr)

    if (present(ny)) then
       ny_local = ny / parallel%size
       if (ny_local * parallel%size /= ny) then
          write(*,*) 'Cannot divide grid evenly to processors'
          call mpi_abort(MPI_COMM_WORLD, -2, ierr)
       end if
    end if

    call mpi_comm_rank(MPI_COMM_WORLD, parallel%rank, ierr)

    parallel%nleft = parallel%rank - 1
    parallel%nright = parallel%rank + 1

    if (parallel%nleft < 0) then
       parallel%nleft = MPI_PROC_NULL
    end if
    if (parallel%nright > parallel%size - 1) then
       parallel%nright = MPI_PROC_NULL
    end if

  end subroutine parallel_initialize

  ! Initialize the temperature field.  Pattern is disc with a radius
  ! of nx_full / 6 in the center of the grid.
  ! Boundary conditions are (different) constant temperatures outside the grid
  subroutine initialize(field0, parallel)
    implicit none

    type(field), intent(inout) :: field0
    type(parallel_data), intent(in) :: parallel

    real(dp) :: radius2
    integer :: i, j, ds2

    ! The arrays for field contain also a halo region
    allocate(field0%data(0:field0%nx+1, 0:field0%ny+1))

    ! Square of the disk radius
    radius2 = (field0%nx_full / 6.0_dp)**2

    do j = 0, field0%ny + 1
       do i = 0, field0%nx + 1
          ds2 = int((i - field0%nx_full / 2.0_dp + 1)**2 + &
               & (j + parallel%rank * field0%ny - field0%ny_full / 2.0_dp + 1)**2)
          if (ds2 < radius2) then
             field0%data(i,j) = 5.0_dp
          else
             field0%data(i,j) = 65.0_dp
          end if
       end do
    end do

    ! Boundary conditions
    if (parallel % rank == 0) then
       field0%data(:,0) = 20.0_dp
    else if (parallel % rank == parallel%size - 1) then
       field0%data(:,field0%ny+1) = 70.0_dp
    end if
    field0%data(0,:) = 85.0_dp
    field0%data(field0%nx+1,:) = 5.0_dp

  end subroutine initialize

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

  ! Exchange the boundary data between MPI tasks
  subroutine exchange(field0, parallel)
    implicit none

    type(field), intent(inout) :: field0
    type(parallel_data), intent(in) :: parallel

    integer :: ierr

    ! TODO: Send to left, receive from right


    ! TODO: Send to right, receive from left


  end subroutine exchange

  ! Output routine, saves the temperature distribution as a png image
  ! Arguments:
  !   curr (type(field)): variable with the temperature data
  !   iter (integer): index of the time step
  subroutine output(curr, iter, parallel)
    use pngwriter
    implicit none

    type(field), intent(in) :: curr
    integer, intent(in) :: iter
    type(parallel_data), intent(in) :: parallel

    character(len=85) :: filename

    integer :: stat
    real(dp), dimension(:,:), allocatable, target :: full_data
    integer :: p, ierr

    if (parallel%rank == 0) then
       allocate(full_data(curr%nx_full, curr%ny_full))
       ! Copy rand #0 data to the global array
       full_data(1:curr%nx, 1:curr%ny) = curr%data(1:curr%nx, 1:curr%ny)

       ! Receive data from other ranks
       do p = 1, parallel%size - 1
          call mpi_recv(full_data(1:curr%nx, p*curr%ny + 1:(p + 1) * curr%ny), &
               & curr%nx * curr%ny, MPI_DOUBLE_PRECISION, p, 22, &
               & MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
       end do
       write(filename,'(A5,I4.4,A4,A)')  'heat_', iter, '.png'
       stat = save_png(full_data, curr%nx_full, curr%ny_full, filename)
       deallocate(full_data)
    else
       ! Send data
       call mpi_send(curr%data(1:curr%nx, 1:curr%ny), curr%nx * curr%ny, MPI_DOUBLE_PRECISION, 0, 22, &
            & MPI_COMM_WORLD, ierr)
    end if

  end subroutine output

  ! Clean up routine for field type
  ! Arguments:
  !   field0 (type(field)): field variable to be cleared
  subroutine finalize(field0)
    implicit none

    type(field), intent(inout) :: field0

    deallocate(field0%data)
  end subroutine finalize

  ! Reads the temperature distribution from an input file
  ! Arguments:
  !   field0 (type(field)): field variable that will store the
  !                         read data
  !   filename (char): name of the input file
  ! Note that this version assumes the input data to be in C memory layout
  subroutine read_input(field0, filename, parallel)
    implicit none

    type(field), intent(out) :: field0
    character(len=85), intent(in) :: filename
    type(parallel_data), intent(out) :: parallel

    integer :: nx, ny, i, ierr
    character(len=2) :: dummy

    real(dp), dimension(:,:), allocatable :: full_data, inner_data

    open(10, file=filename)
    ! Read the header
    read(10, *) dummy, nx, ny

    call parallel_initialize(parallel, nx, ny)
    call initialize_field_metadata(field0, nx, ny, parallel)

    ! The arrays for temperature field contain also a halo region
    allocate(field0%data(0:field0%nx+1, 0:field0%ny+1))

    allocate(inner_data(field0%nx, field0%ny))

    if (parallel%rank == 0) then
       allocate(full_data(nx, ny))
       ! Read the data
       do i = 1, nx
          read(10, *) full_data(i, 1:ny)
       end do
    end if

    call mpi_scatter(full_data, nx * field0%ny, MPI_DOUBLE_PRECISION, inner_data, &
         & nx * field0%ny, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)
    ! Copy to full array containing also boundaries
    field0%data(1:field0%nx, 1:field0%ny) = inner_data(:,:)

    ! Set the boundary values
    field0%data(1:field0%nx, 0) = field0%data(1:field0%nx, 1)
    field0%data(1:field0%nx, field0%ny + 1) = field0%data(1:field0%nx, field0%ny)
    field0%data(0, 0:field0%ny + 1) = field0%data(1, 0:field0%ny + 1)
    field0%data(field0%nx + 1, 0:field0%ny + 1) = field0%data(field0%nx, 0:field0%ny + 1)

    close(10)
    deallocate(inner_data)
    if (parallel%rank == 0) then
       deallocate(full_data)
    end if

  end subroutine read_input

end module heat_mpi
