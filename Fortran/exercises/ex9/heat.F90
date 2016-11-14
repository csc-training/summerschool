
module heat
  use iso_fortran_env, only : REAL64
  implicit none

  integer, parameter :: dp = REAL64

  real(dp), parameter :: DX = 0.01, DY = 0.01  ! Fixed grid spacing

  type :: field
     integer :: nx
     integer :: ny
     real(dp) :: dx
     real(dp) :: dy
     real(dp), dimension(:,:), allocatable :: data
  end type field

contains

  ! Initialize the field type metadata
  ! Arguments:
  !   field0 (type(field)): input field
  !   nx, ny, dx, dy: field dimensions and spatial step size
  subroutine initialize_field_metadata(field0, nx, ny)
    implicit none

    type(field), intent(out) :: field0
    integer, intent(in) :: nx, ny

    field0%dx = DX
    field0%dy = DY
    field0%nx = nx
    field0%ny = ny
  end subroutine initialize_field_metadata

  ! Initialize the temperature field.  Pattern is disc 
  ! Boundary conditions are (different) constant temperatures outside the grid
  subroutine initialize(field0)
    implicit none

    type(field), intent(inout) :: field0

    real(dp) :: radius2
    integer :: i, j, ds2

    ! The arrays for field contain also a halo region
    allocate(field0%data(0:field0%nx+1, 0:field0%ny+1))

    ! Square of the disk radius
    radius2 = (field0%nx / 6.0)**2

    do j = 0, field0%ny + 1
       do i = 0, field0%nx + 1
          ds2 = int((i - field0%nx / 2.0 + 1)**2 + &
               & (j - field0%ny / 2.0 + 1)**2)
          if (ds2 < radius2) then
             field0%data(i,j) = 5.0
          else
             field0%data(i,j) = 65.0
          end if
       end do
    end do

    ! Boundary conditions
    do j = 0, field0%nx + 1
       field0%data(j, 0) = real(20, dp)
    end do

    do j = 0, field0%nx + 1
       field0%data(j, field0%ny + 1) = real(70, dp)
    end do

    do j = 0, field0%ny + 1
       field0%data(0, j) = real(85, dp)
    end do

    do j = 0, field0%ny+1
       field0%data(field0%nx + 1, j) = real(5, dp)
    end do
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
! TODO: implement this.
  end subroutine evolve

  ! Output routine, saves the temperature distribution as a png image
  ! Arguments:
  !   curr (type(field)): variable with the temperature data
  !   iter (integer): index of the time step
  subroutine output(curr, iter)
    use pngwriter
    implicit none

    type(field), intent(in) :: curr
    integer, intent(in) :: iter
    character(len=85) :: filename

    ! The actual write routine takes only the actual data
    ! (without ghost layers) so we need array for that
    integer :: full_nx, full_ny, stat
    real(dp), dimension(:,:), allocatable, target :: full_data

    full_nx = curr%nx
    full_ny = curr%ny

    allocate(full_data(full_nx, full_ny))
    full_data(1:curr%nx, 1:curr%ny) = curr%data(1:curr%nx, 1:curr%ny)

    write(filename,'(A5,I4.4,A4,A)')  'heat_', iter, '.png'
    stat = save_png(full_data, full_nx, full_ny, filename)
    deallocate(full_data)
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
  subroutine read_input(field0, filename)
    implicit none

    type(field), intent(out) :: field0
    character(len=85), intent(in) :: filename

    integer :: nx, ny, i
    character(len=2) :: dummy

    open(10, file=filename)
! TODO: implement this routine.
    ! Read the header

    ! The arrays for temperature field contain also a halo region


    ! Read the data


    ! Set the boundary values

  end subroutine read_input

end module heat
