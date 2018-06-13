module io

contains

  ! Reads the temperature distribution from an input file
  subroutine read_field(field, filename)
    implicit none

    real, dimension(:,:), allocatable, intent(out) :: field
    character(len=*), intent(in) :: filename

    integer, parameter :: funit = 10
    integer :: nx, ny, i, stat, alloc_stat
    character(len=2) :: dummy

    ! TODO: implement function that will:
    ! open the file
    ! read the first header line to get nx and ny
    ! allocate matrix called field
    ! read rest of the file into field
    ! close the file
    open(funit, file=filename, status='old', iostat=stat)
    if (stat /= 0) then
       write (*,*) 'Error, could not open ' // trim(filename)
       stop
    end if

    ! read header
    read(funit, *) dummy, nx, ny

    ! allocate matrix
    if (allocated(field)) deallocate(field)
    allocate(field(1:nx, 1:ny), stat=alloc_stat)
    if (alloc_stat /= 0) call abort()

    ! read data into inner regions
    do i = 1, nx
       read(funit, *) field(i, 1:ny)
    end do

    close(funit)
  end subroutine read_field

  ! Output routine, saves the temperature distribution as a png image
  subroutine write_field(field, iter)
    use iso_fortran_env, only : REAL64
    use pngwriter
    implicit none

    integer, parameter :: dp = REAL64
    real, intent(in) :: field(:,:)
    integer, intent(in) :: iter

    character(len=85) :: filename
    integer :: nx, ny, stat

    nx = size(field, 1)
    ny = size(field, 2)


    write(filename,'(A5,I4.4,A4,A)')  'heat_', iter, '.png'
    stat = save_png(real(field, kind=dp), nx, ny, filename)
    if (stat == 0) then
       write (*,*) 'Wrote the png file ', filename
       write (*,*) 'Use e.g. "eog" to open it.'
    end if
  end subroutine write_field

end module io
