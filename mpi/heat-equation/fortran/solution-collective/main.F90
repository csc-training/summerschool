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
  integer, parameter :: image_interval = 100 ! Image output interval

  type(parallel_data) :: parallelization
  integer :: ierr

  integer :: iter

  real(dp) :: average_temp   !  Average temperature

  real(kind=dp) :: start, stop ! Timers

  call mpi_init(ierr)

  call initialize(current, previous, nsteps, parallelization)

  ! Draw the picture of the initial state
  call write_field(current, 0, parallelization)

  average_temp = average(current, parallelization)
  if (parallelization % rank == 0) then
     write(*,'(A, I5, A, I5, A, I5)') 'Simulation grid: ', current%nx_full, ' x ', & 
          & current%ny_full, ' time steps: ', nsteps
     write(*,'(A, I5)') 'MPI processes: ', parallelization%size
     write(*,'(A,F9.6)') 'Average temperature at start: ', average_temp
  end if

  ! Largest stable time step
  dt = current%dx**2 * current%dy**2 / &
       & (2.0 * a * (current%dx**2 + current%dy**2))

  ! Main iteration loop, save a picture every
  ! image_interval steps

  start =  mpi_wtime()

  do iter = 1, nsteps
     call exchange(previous, parallelization)
     call evolve(current, previous, a, dt)
     if (mod(iter, image_interval) == 0) then
        call write_field(current, iter, parallelization)
     end if
     call swap_fields(current, previous)
  end do

  stop = mpi_wtime()

  ! Average temperature for reference
  average_temp = average(previous, parallelization)

  if (parallelization % rank == 0) then
     write(*,'(A,F7.3,A)') 'Iteration took ', stop - start, ' seconds.'
     write(*,'(A,F9.6)') 'Average temperature: ',  average_temp
     if (command_argument_count() == 0) then
         write(*,'(A,F9.6)') 'Reference value with default arguments: ', 59.281239
     end if
  end if

  call finalize(current, previous)

  call mpi_finalize(ierr)

end program heat_solve
