! Heat equation solver in 2D.

program heat_solve
  use heat
  use core
  use io
  use setup
  use utilities
  use mpi
  use omp_lib

  implicit none

  real(dp), parameter :: a = 0.5 ! Diffusion constant
  type(field) :: current, previous    ! Current and previus temperature fields

  real(dp) :: dt     ! Time step
  integer :: nsteps       ! Number of time steps
  integer, parameter :: image_interval = 500 ! Image output interval

  type(parallel_data) :: parallelization
  integer :: ierr, provided, thread_id

  integer :: iter

  real(dp) :: start, stop ! Timers

  call mpi_init_thread(MPI_THREAD_MULTIPLE,provided,ierr)
  if (provided < MPI_THREAD_MULTIPLE) then
     write (*,*) ' MPI_THREAD_MULTIPLE required for the thread support level '
     call mpi_abort(mpi_comm_world, 5, ierr)
  end if
!$OMP PARALLEL PRIVATE(iter, thread_id)
  call initialize(current, previous, nsteps, parallelization)
  thread_id = omp_get_thread_num()

  ! Draw the picture of the initial state
!$OMP SINGLE
  call write_field(current, 0, parallelization)

  ! Largest stable time step
  dt = current%dx**2 * current%dy**2 / &
       & (2.0 * a * (current%dx**2 + current%dy**2))

  ! Main iteration loop, save a picture every
  ! image_interval steps
!$OMP END SINGLE
  start =  mpi_wtime()  
  do iter = 1, nsteps
     call exchange(previous, parallelization, thread_id)
     call evolve(current, previous, a, dt)
!$OMP SINGLE
     if (mod(iter, image_interval) == 0) then
        call write_field(current, iter, parallelization)
     end if
     call swap_fields(current, previous)
!$OMP END SINGLE
  end do
  stop = mpi_wtime()
!$OMP END PARALLEL

  if (parallelization % rank == 0) then
     write(*,'(A,F7.3,A)') 'Iteration took ', stop - start, ' seconds.'
     write(*,'(A,G0)') 'Reference value at 5,5: ', previous % data(5,5)
  end if
  
  call finalize(current, previous)

  call mpi_finalize(ierr)

end program heat_solve
