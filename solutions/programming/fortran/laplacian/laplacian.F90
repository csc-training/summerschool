program laplacian
  use iso_fortran_env, only : REAL64
  implicit none
  integer, parameter :: dp = REAL64

  integer, parameter :: nx = 16, ny = 16
  real(dp), dimension(nx, ny) :: prev, lapl
  integer :: i, j

  real(dp), parameter :: dx = 0.01, dy = 0.01



! initialize prev array with varying boundaries
  prev(:,:)  = 0.0 ! center
  prev(1,:)  = 3.0 ! left
  prev(nx,:) = 4.0 ! right
  prev(:,1)  = 1.0 ! top
  prev(:,ny) = 2.0 ! bottom

! initialize lapl array to zeros
  lapl(:,:)  = 0.0 ! center

!-------------------------------------------------- 


  ! Compute Laplacian in double do-loop using prev 
  ! and saving to lapl array.
  do j = 2, ny-1
    do i = 2, nx-1
      lapl(i,j) = (prev(i-1, j) - 2.0 * prev(i,j) + &
           &       prev(i+1, j)) /dx**2 + &
           &      (prev(i, j-1) - 2.0 * prev(i,j) + &
           &       prev(i, j+1)) /dy**2
    end do
  end do




!--------------------------------------------------
! printing of the prev and lapl arrays
  write(*,*) "Previous array:"
  do j = 1, ny
     write(*,'(*(F4.1))') prev(:,j)
  end do

  write(*,*) "Laplacian of the array:"
  do j = 1, ny
     write(*,'(*(F4.1))') lapl(:,j)
  end do

end program laplacian
