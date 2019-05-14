program mandelbrot

  use iso_fortran_env, only : REAL64
  use pngwriter
  use omp_lib

  implicit none

  integer, parameter :: dp = REAL64

  integer, parameter :: max_iter_count = 512
  integer, parameter :: max_depth = 6
  integer, parameter :: min_size = 32
  integer, parameter :: subdiv = 4

  integer, parameter :: w = 2048
  integer, parameter :: h = w

  integer, pointer, dimension(:,:) :: iter_counts

  integer :: stat

  complex(dp) :: cmin, cmax
  real(dp) :: t0, t1

  allocate(iter_counts(w, h))


  t0 = omp_get_wtime()

  cmin = (-1.5, -1.0)
  cmax = (0.5, 1.0)

  !$omp parallel
  !$omp single
  call mandelbrot_block(iter_counts, w, h, cmin, cmax, 0, 0, w, 1)
  !$omp end single
  !$omp end parallel

  t1 = omp_get_wtime()

  stat = save_png(iter_counts, h, w, 'mandelbrot.png')

  deallocate(iter_counts)

  write(*,*) 'Mandelbrot set computed in', t1 - t0, 's'

contains

  ! Computes the Mandelbrot image recursively
  ! At each call, the image is divided into smaller blocks (by a factor of
  ! subdiv), and the function is called recursively with arguments corresponding
  ! to subblock. When maximum recursion depth is reached or size of block
  ! is smaller than predefined minimum, one starts to calculate actual pixel
  ! values
  !
  ! - - - - - - - -           -----       -----
  ! |             |           |   |       |   |
  ! |             |           -----       -----
  ! |             |  -->                         -->   ...
  ! |             |           -----       -----
  ! |             |           |   |       |   |
  ! |             |           -----       -----
  ! ---------------
  !

  recursive subroutine mandelbrot_block(iter_counts, w, h, cmin, cmax, &
       x0, y0, d, depth)
    implicit none

    integer, pointer, dimension(:,:), intent(inout) :: iter_counts
    integer, intent(in) :: w, h, x0, y0, d, depth
    complex(dp), intent(in) :: cmin, cmax

    integer :: block_size, i, j

    block_size = d / subdiv
    if ((depth + 1 < max_depth) .and. (block_size > min_size)) then
       ! Subdivide recursively
       do i=0, subdiv - 1
          do j=0, subdiv - 1
             !$omp task
             call mandelbrot_block(iter_counts, w, h, cmin, cmax, &
                  x0 + i*block_size, y0 + j*block_size, &
                  block_size, depth + 1)
             !$omp end task
          end do
       end do
    else
       ! Last recursion level reached, calculate the values
       do j = y0 + 1, y0 + d
          do i = x0 + 1, x0 + d
             iter_counts(i, j) = kernel(h, w, cmin, cmax, i-1, j-1)
          end do
       end do
    end if

  end subroutine mandelbrot_block

  ! Calculate iteration count for a pixel.
  ! This function does not need to be edited
  integer function kernel(h, w, cmin, cmax, x, y)
    implicit none

    integer, intent(in) :: h, w, x, y
    complex(dp), intent(in) :: cmin, cmax

    integer :: iteration
    complex(dp) :: z, dc, c
    real(dp) :: fx, fy

    dc = cmax - cmin
    fx = real(x, dp) / w
    fy = real(y, dp) / h
    c = cmin + fx * real(dc) +  fy * imag(dc) * (0.0, 1.0)

    z = c
    iteration = 0

    do while (iteration < max_iter_count .and. abs(z)**2 < 4)
       z = z**2 + c
       iteration = iteration + 1
    end do

    kernel = iteration

  end function kernel

end program mandelbrot
