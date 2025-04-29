program mandelbrot
  use constants
  use mandelbrot_kernel
  use pngwriter
  use omp_lib
  implicit none

  integer, parameter :: num_blocks = 8
  integer :: block_size, y_start, y_end, x, y, block
  integer, dimension(0:width-1, 0:height-1) :: image
  real(dp) :: t0, t1

  integer :: stat

  t0 = omp_get_wtime()

  !$omp target data map(to:image(0:width-1, 0:height-1))
  do block = 0, num_blocks - 1
    y_start = block * (height / num_blocks)
    y_end = y_start + (height / num_blocks)
    !$omp target teams distribute parallel do collapse(2) &
    !$omp depend(out:image(0, y_start)) nowait
    do y = y_start, y_end - 1
      do x = 0, width - 1
        image(x, y) = kernel(x, y)
      end do
    end do
    !$omp end target teams distribute parallel do

    !$omp target update from(image(0:width-1, y_start:y_end-1)) &
    !$omp depend(in:image(0, y_start)) nowait
  end do

  !$omp end target data

  !$omp taskwait

  t1 = omp_get_wtime()

  stat = save_png(image, height, width, 'mandelbrot.png')

  write(*,*) 'Time spent', t1 - t0

end program
