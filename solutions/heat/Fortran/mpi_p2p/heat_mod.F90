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
     integer :: nleft, nright  ! Ranks of neighbouring MPI tasks
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

    nx_local = nx
    ny_local = ny / parallel%size

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
    
  end subroutine parallel_setup

end module heat
