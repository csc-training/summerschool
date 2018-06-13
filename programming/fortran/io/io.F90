module io

contains

  ! Reads the temperature distribution from an input file
  subroutine read_field(field, filename)
    implicit none

    real, dimension(:,:), allocatable, intent(out) :: field
    character(len=*), intent(in) :: filename

    ! TODO: implement function that will:
    ! open the file
    ! read the first header line to get nx and ny
    ! allocate matrix called field
    ! read rest of the file into field
    ! close the file

















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
