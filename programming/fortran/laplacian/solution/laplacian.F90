program laplacian
  use iso_fortran_env, only : REAL64
  implicit none

  integer, parameter :: nx = 10, ny = 10, dp = REAL64
  real(dp), dimension(nx,ny) :: U, lapl, solution
  integer :: i, j

  real(dp) :: dx, dy, x, y

  ! Grid spacing
  dx = 1.0/real(nx)
  dy = 1.0/real(ny)

  ! TODO: 
  ! initialize array U(x,y) = (x^2 + y^2) in the domain [0:1,0:1] 
  y = 0.0
  do j = 1, ny
     x = 0.0
     do i = 1, nx
        U(i,j) =  x**2 + y**2
        x = x + dx
     end do
     y = y + dy
  end do

  ! TODO:
  ! Compute Laplacian of U in double do-loop and saving to lapl array.
  lapl = 0.0
  do j = 2, ny-1
    do i = 2, nx-1
       lapl(i,j) = (U(i-1,j) - 2.0*U(i,j) + U(i+1,j)) / dx**2 + &
            (U(i,j-1) - 2.0*U(i,j) + U(i,j+1)) / dy**2
    end do
  end do


  !--------------------------------------------------
  ! TODO:
  ! Printing of the prev and lapl arrays
  write(*,*) "Original array:"
  do i = 2, nx-1
    write(*,'(*(G9.1))') U(i,2:ny-1)
  end do

  write(*,*) "Laplacian of the array:"
  do i = 2, nx-1
    write(*,'(*(G9.1))') lapl(i,2:ny-1)
  end do

 ! Analytically, the Laplacian of the function is nabla^2 U(x,y) = 4


end program laplacian
