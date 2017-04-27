! Heat equation solver in 2D.

program heat_solve
  use heat
  use core
  use io
  use setup
  use utilities

  implicit none

  real(dp), parameter :: a = 0.5 ! Diffusion constant
  type(field) :: current, previous    ! Current and previus temperature fields

  real(dp) :: dt     ! Time step
  integer :: nsteps       ! Number of time steps
  integer, parameter :: image_interval = 10 ! Image output interval

  integer :: iter

  real :: start, stop ! Timers

  call initialize(current, previous, nsteps)

  ! Draw the picture of the initial state
  call write_field(current, 0)

  ! Largest stable time step
  dt = current%dx**2 * current%dy**2 / &
       & (2.0 * a * (current%dx**2 + current%dy**2))

  ! Main iteration loop, save a picture every
  ! image_interval steps

  call cpu_time(start)
  
  do iter = 1, nsteps
     call evolve(current, previous, a, dt)
     if (mod(iter, image_interval) == 0) then
        call write_field(current, iter)
     end if
     call swap_fields(current, previous)
  end do

  call cpu_time(stop)

  write(*,'(A,F7.3,A)') 'Iteration took ', stop - start, ' seconds.'
  write(*,'(A,G0)') 'Reference value at 5,5: ', previous % data(5,5)
  
  call finalize(current, previous)

end program heat_solve
