module laplacian_mod
  implicit none
  real, parameter :: dx = 0.01, dy = 0.01

contains

  subroutine initialize(field0)
    implicit none
    real, dimension(0:,0:), intent(inout) :: field0
    integer :: lbx, lby, ubx, uby

    ! lbound and ubound return the lower and upper boundaries (indices) of an array, respectively
    lbx = lbound(field0,1)
    ubx = ubound(field0,1)
    lby = lbound(field0,2)
    uby = ubound(field0,2)

    ! initialize middle
    field0 = 65.0

    ! or use random values to better see effects from applying laplacian
    !call random_number(field0)

    ! Adjust the boundaries
    field0(:, lby) = 20.0
    field0(:, uby) = 70.0
    field0(lbx, :) = 85.0
    field0(ubx, :) = 5.0
  end subroutine initialize

  subroutine laplacian(curr, prev)
    implicit none
    real, dimension(0:,0:), intent(in) :: prev
    real, dimension(0:,0:), intent(out) :: curr
    real, parameter :: dx = 0.01, dy = 0.01
    integer :: nx, ny, i, j

    nx = size(prev,1)
    ny = size(prev,2)

    ! Compute Laplacian in double do-loop using "prev" array
    ! and saving the result to "curr" array.
    do j = 1, ny-2
       do i = 1, nx-2
          curr(i,j) = (prev(i-1, j) - 2.0 * prev(i,j) + &
               &       prev(i+1, j)) /dx**2 + &
               &      (prev(i, j-1) - 2.0 * prev(i,j) + &
               &       prev(i, j+1)) /dy**2
       end do
    end do
  end subroutine laplacian

  subroutine write_field(array)
    implicit none
    real, dimension(0:,0:), intent(in) :: array
    integer :: i, nx, ny

    nx = size(array,1)
    ny = size(array,2)
    write(*,*) ' '
    do i = 1, nx-2
       write(*,'(*(G10.1))') array(i,1:ny-2)
    end do

  end subroutine write_field

  subroutine swap_fields(curr, prev)
    implicit none
    real, allocatable, dimension(:,:), intent(inout) :: curr, prev
    real, allocatable, dimension(:,:) :: tmp

    ! introducing a handy array intrinsic for allocating and copying and array in one go: move_alloc
    call move_alloc(curr, tmp) ! tmp = curr
    call move_alloc(prev, curr) ! curr = prev
    call move_alloc(tmp, prev) ! prev = tmp
  end subroutine swap_fields


end module laplacian_mod
