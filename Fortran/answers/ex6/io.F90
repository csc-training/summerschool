program io
  implicit none
  
  integer, parameter :: dp = selected_real_kind(12)
  real(kind=dp), dimension(:,:), allocatable :: field

  call read_input(field, 'bottle.dat')
  call output(field)

contains

  subroutine read_input(field, filename)
    implicit none

    real(kind=dp), dimension(:,:), allocatable, intent(out) :: field
    character(len=*), intent(in) :: filename

    integer, parameter :: funit = 10
    integer :: nx, ny, i, stat, alloc_stat
    character(len=2) :: dummy

    open(funit, file=filename, status='old', iostat=stat)
    if (stat /= 0) then
      write (*,*) 'Error, could not open ' // trim(filename)
      stop
    end if

    ! read header 
    read(funit, *) dummy, ny, nx

    ! allocate matrix
    if (allocated(field)) deallocate(field)
    allocate(field(1:nx, 1:ny), stat=alloc_stat)
    if (alloc_stat /= 0) call abort()

    ! read data into inner regions
    do i = 1, nx
      read(funit, *) field(i, 1:ny)
    end do

    close(funit)
  end subroutine read_input

  subroutine output(fld)
    use, intrinsic :: ISO_C_BINDING
    implicit none

    real(kind=dp), intent(in) :: fld(:,:)
    integer :: nx, ny, stat

    ! Interface for save_png C-function
    interface
       ! The C-function definition is
       !   int save_png(double *data,
       !                const int nx, const int ny,
       !                const char *fname, const char lang)
       function save_png(data, nx, ny, fname, lang) &
            & bind(C,name="save_png") result(stat)
         use, intrinsic :: ISO_C_BINDING
         implicit none
         real(kind=C_DOUBLE) :: data(*)
         integer(kind=C_INT), value, intent(IN) :: nx, ny
         character(kind=C_CHAR), intent(IN) :: fname(*)
         character(kind=C_CHAR), value, intent(IN) :: lang
         integer(kind=C_INT) :: stat
       end function save_png
    end interface   

    nx = size(fld, 1)
    ny = size(fld, 2)

    stat = save_png(field, nx, ny, &
         & 'ex6.png' // C_NULL_CHAR, 'F')
    if (stat == 0) write(*,*) 'Output written to ex6.png'
  end subroutine output
end program io
