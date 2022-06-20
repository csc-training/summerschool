! PNG writer
module pngwriter

contains

  function save_png(data, nx, ny, fname) result(stat)

    use, intrinsic :: ISO_C_BINDING
    implicit none

    integer, dimension(:,:), intent(in) :: data
    integer, intent(in) :: nx, ny
    character(len=*), intent(in) :: fname
    integer :: stat

    ! Interface for save_png C-function
    interface
       ! The C-function definition is
       !   int save_png(double *data, const int nx, const int ny,
       !                const char *fname)
       function save_png_c(data, nx, ny, fname, order) &
            & bind(C,name="save_png") result(stat)
         use, intrinsic :: ISO_C_BINDING
         implicit none
         integer(kind=C_INT) :: data(*)
         integer(kind=C_INT), value, intent(IN) :: nx, ny
         character(kind=C_CHAR), intent(IN) :: fname(*)
         character(kind=C_CHAR), value, intent(IN) :: order
         integer(kind=C_INT) :: stat
       end function save_png_c
    end interface

    stat = save_png_c(data, nx, ny, trim(fname) // C_NULL_CHAR, 'f')
    if (stat /= 0) then
       write(*,*) 'save_png returned error!'
    end if

  end function save_png

end module pngwriter
