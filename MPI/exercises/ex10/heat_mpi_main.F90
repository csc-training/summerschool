
program heat_solve
  use heat_mpi
  use mpi
  implicit none

  real(kind=dp), parameter :: a = 0.5 ! Diffusion constant
  type(field) :: current, previous    ! Current and previus temperature fields

  real(kind=dp) :: dt     ! Time step
  integer :: nsteps       ! Number of time steps
  integer, parameter :: image_interval = 10 ! Image output interval

  integer :: rows, cols  ! Field dimensions

  character(len=85) :: input_file, arg  ! Input file name and command line arguments

  type(parallel_data) :: parallelization
  integer :: ierr

  integer :: iter, step0

  logical :: using_input_file

  real(kind=dp) :: start, stop

  call mpi_init(ierr)

  ! Default values for grid size and time steps
  rows = 200
  cols = 200
  nsteps = 500
  using_input_file = .false.

  step0 = 0
  ! Read in the command line arguments and
  ! set up the needed variables
  select case(command_argument_count())
  case(0) ! No arguments -> default values
  case(1) ! One argument -> input file name
     using_input_file = .true.
     call get_command_argument(1, input_file)
  case(2) ! Two arguments -> input file name and number of steps
     using_input_file = .true.
     call get_command_argument(1, input_file)
     call get_command_argument(2, arg)
     read(arg, *) nsteps
  case(3) ! Three arguments -> rows, cols and nsteps
     call get_command_argument(1, arg)
     read(arg, *) rows
     call get_command_argument(2, arg)
     read(arg, *) cols
     call get_command_argument(3, arg)
     read(arg, *) nsteps
  case default
     call usage()
     stop
  end select
  ! Initialize the fields according the command line arguments
  if (using_input_file) then
     call get_command_argument(1, input_file)
     call read_input(previous, input_file, parallelization)
     call copy_fields(previous, current)
  else
     call parallel_initialize(parallelization, rows, cols)
     call initialize_field_metadata(previous, rows, cols, parallelization)
     call initialize_field_metadata(current, rows, cols, parallelization)
     call initialize(previous, parallelization)
     call initialize(current, parallelization)
  end if
  ! Draw the picture of the initial state
  call output(current, 0, parallelization)

  ! Largest stable time step
  dt = current%dx**2 * current%dy**2 / &
       & (2.0 * a * (current%dx**2 + current%dy**2))

  ! Main iteration loop, save a picture every
  ! image_interval steps

  start = mpi_wtime()
  
  do iter = 1, nsteps
     call exchange(previous, parallelization)
     call evolve(current, previous, a, dt)
     if (mod(iter, image_interval) == 0) then
        call output(current, iter, parallelization)
     end if
     call swap_fields(current, previous)
  end do

  stop = mpi_wtime()

  if (parallelization % rank == 0) then
     write(*,'(A,F7.3,A)') 'Iteration took ', stop - start, ' seconds.'
     write(*,'(A,G0)') 'Reference value at 5,5: ', previous % data(5,5)
  end if
  
  call finalize(current)
  call finalize(previous)
  call mpi_finalize(ierr)

contains

  ! Helper routine that prints out a simple usage if
  ! user gives more than three arguments
  subroutine usage()
    implicit none
    character(len=256) :: buf

    call get_command_argument(0, buf)
    write (*,'(A)') 'Usage:'
    write (*,'(A, " (default values will be used)")') trim(buf)
    write (*,'(A, " <filename>")') trim(buf)
    write (*,'(A, " <filename> <nsteps>")') trim(buf)
    write (*,'(A, " <rows> <cols> <nsteps>")') trim(buf)
  end subroutine usage

end program heat_solve
