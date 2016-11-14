module laplacian_mod
  use iso_fortran_env, only : REAL64
  implicit none
  integer, parameter :: dp = REAL64

  type :: field
     integer :: nx
     integer :: ny
     real(dp) :: dx = 0.01
     real(dp) :: dy = 0.01
     real(dp), dimension(:,:), allocatable :: data
  end type field

contains

  subroutine initialize
    ! TODO
  end subroutine initialize

  subroutine laplacian
    ! TODO
  end subroutine laplacian

  subroutine print_field(field0)
    implicit none
    type(field), intent(in) :: field0
    integer :: i

    do i = 1, field0%nx
       write(*,'(*(G6.1))') field0%data(i,:)
    end do
  end subroutine print_field
  
end module laplacian_mod
