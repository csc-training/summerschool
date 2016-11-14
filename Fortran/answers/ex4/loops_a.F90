program loops
  implicit none
  integer, parameter :: nx = 16, ny = 16, dist = 16
  real, parameter :: dx = 0.01, dy = 0.01
  real, dimension(0:ny+1,0:nx+1) :: field, field_new
  real :: d
  integer :: i, j

  do j = 0, nx + 1
     do i = 0, ny + 1
        if ((i+j) < dist) then
           field(i,j) = 5.0
        else
           field(i,j) = 1.0
        end if
     end do
  end do


  field_new = 0.0
  call apply_fd_laplacian(field, field_new, dx, dy)

  if (check_result(field, field_new, dx, dy)) then
     write (*,*) 'The result seems ok'
  else
     write (*,*) 'The result does not seem ok'
  end if


  contains
    subroutine apply_fd_laplacian(x, y, dx, dy)
      implicit none
      real, intent(in) :: x(:,:)
      real, intent(out) :: y(:,:)
      real, intent(in) :: dx, dy
      
      real :: dx2, dy2
      integer :: i, j, lnx, unx, lny, uny

      lnx = lbound(x,1)
      unx = ubound(x,1)
      lny = lbound(x,2)
      uny = ubound(x,2)      
      dx2 = dx**2
      dy2 = dy**2

      do j=lny+1,uny-1
         do i=lnx+1,unx-1
            y(i,j)=(x(i-1,j)-2*x(i,j)+x(i+1,j))/dx2+ &
                 &  (x(i,j-1)-2*x(i,j)+x(i,j+1))/dy2
         end do
      end do
    end subroutine apply_fd_laplacian

    function check_result(x, y, dx, dy) result(iscorrect)
      implicit none
      real, intent(in) :: x(:,:), y(:,:)
      real, intent(in) :: dx, dy

      real :: yc(size(x,1),size(x,2)) 
      integer :: lny, uny, lnx, unx
      real, parameter :: tol = 1e-15
      logical :: iscorrect 

      lny = lbound(x,2)+1
      uny = ubound(x,2)-1
      lnx = lbound(x,1)+1
      unx = ubound(x,1)-1

      yc = 0.0

      yc(lnx:unx,lny:uny) = &
           & (x(lnx-1:unx-1,lny:uny)-2*x(lnx:unx,lny:uny)+x(lnx+1:unx+1,lny:uny))/dx**2 + &
           & (x(lnx:unx,lny-1:uny-1)-2*x(lnx:unx,lny:uny)+x(lnx:unx,lny+1:uny+1))/dy**2

      iscorrect = all(abs(yc(lnx:unx,lny:uny)-y(lnx:unx,lny:uny))<tol)
    end function check_result

end program loops
