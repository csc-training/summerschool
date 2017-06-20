! Field metadata for heat equation solver
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
  subroutine set_field_dimensions(field0, nx, ny)

    implicit none

    type(field), intent(out) :: field0
    integer, intent(in) :: nx, ny

    field0%dx = DX
    field0%dy = DY
    field0%nx = nx
    field0%ny = ny
  end subroutine set_field_dimensions

end module heat
