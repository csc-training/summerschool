! I/O routines for heat equation solver
module io

contains

  ! Output routine, saves the temperature distribution as a png image
  ! Arguments:
  !   curr (type(field)): variable with the temperature data
  !   iter (integer): index of the time step
  subroutine write_field(curr, iter)
    use heat
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
  end subroutine write_field


  ! Reads the temperature distribution from an input file
  ! Arguments:
  !   field0 (type(field)): field variable that will store the
  !                         read data
  !   filename (char): name of the input file
  ! Note that this version assumes the input data to be in C memory layout
  subroutine read_field(field0, filename)
    use heat

    implicit none

    type(field), intent(out) :: field0
    character(len=85), intent(in) :: filename

    integer :: nx, ny, i
    character(len=2) :: dummy

    open(10, file=filename)
    ! Read the header
    read(10, *) dummy, nx, ny ! nx is the number of rows

    call set_field_dimensions(field0, nx, ny)
    ! The arrays for temperature field contain also a halo region
    allocate(field0%data(0:field0%nx+1, 0:field0%ny+1))

    ! Read the data
    do i = 1, nx
       read(10, *) field0%data(i, 1:ny)
    end do

    ! Set the boundary values
    field0%data(1:nx,   0     ) = field0%data(1:nx, 1     )
    field0%data(1:nx,     ny+1) = field0%data(1:nx,   ny  )
    field0%data(0,      0:ny+1) = field0%data(1,    0:ny+1)
    field0%data(  nx+1, 0:ny+1) = field0%data(  nx, 0:ny+1)

    close(10)
  end subroutine read_field

end module io
