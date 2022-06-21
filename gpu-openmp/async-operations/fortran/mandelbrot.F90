program mandelbrot
  use constants
  use mandelbrot_kernel
  use pngwriter
  use omp_lib
  implicit none

  integer, parameter :: num_blocks = 1
  integer :: block_size, y_start, y_end, x, y, block
  integer, allocatable :: image(:,:)
  real(dp) :: t0, t1

  integer :: stat

  allocate(image(0:height-1, 0:width-1))

  t0 = omp_get_wtime()

  do block = 0, num_blocks - 1
    y_start = block * (height / num_blocks)
    y_end = y_start + (height / num_blocks)
    do y = y_start, y_end - 1
      do x = 0, width - 1
        image(x, y) = kernel(x, y)
      end do
    end do
  end do

  t1 = omp_get_wtime()

  stat = save_png(image, height, width, 'mandelbrot.png')

  write(*,*) 'Time spent', t1 - t0

  deallocate(image)

end program
