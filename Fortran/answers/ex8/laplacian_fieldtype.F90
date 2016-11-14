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

  subroutine initialize(field0)
    implicit none
    type(field), intent(inout) :: field0
    integer :: dist
    integer :: i, j

    allocate (field0%data(0:field0%nx+1, 0:field0%ny+1))
    dist = field0%nx
    do j = 0, field0%ny + 1
       do i = 0, field0%nx + 1
          if ((i+j) < dist) then
             field0%data(i,j) = 50.0_dp
          else
             field0%data(i,j) = 10.0_dp
          end if
       end do
    end do
  end subroutine initialize

  subroutine laplacian(field0, field1)
    implicit none
    type(field) :: field0
    type(field) :: field1
    integer :: i, j
    real(dp) :: dx2, dy2

    dx2 = field0%dx**2
    dy2 = field0%dy**2
    do j = 1, field0%ny
       do i = 1, field0%nx
          field1%data(i,j) = ( field0%data(i-1,j) - 2.0*field0%data(i,j) + &
               & field0%data(i+1,j)) / dx2 + &
               & ( field0%data(i,j-1) - 2.0*field0%data(i,j) + &
               & field0%data(i,j+1)) / dy2
       end do
    end do
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
