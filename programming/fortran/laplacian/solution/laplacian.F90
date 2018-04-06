program laplacian
  implicit none

  integer, parameter :: nx = 10, ny = 10
  real, dimension(nx, ny) :: prev, lapl
  integer :: i, j

  real, parameter :: dx = 0.01, dy = 0.01



  ! initialize prev array with varying boundaries
  prev(:,:)  = 65.0 ! middle
  prev(:,1)  = 20.0 ! left
  prev(:,ny) = 70.0 ! right
  prev(1,:)  = 85.0 ! top
  prev(nx,:) = 5.0  ! bottom

  ! initialize lapl array to zeros
  lapl(:,:)  = 0.0  ! middle


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
  ! Printing of the prev and lapl arrays
  write(*,*) "Previous array:"
  do i = 1, nx
    write(*,'(*(G10.1))') prev(i,:)
  end do

  write(*,*) "Laplacian of the array:"
  do i = 1, nx
    write(*,'(*(G10.1))') lapl(i,:)
  end do

end program laplacian
