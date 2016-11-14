program heat_solve
  use heat
  !$ use omp_lib
  implicit none

  real(kind=dp), parameter :: a = 0.5 ! Diffusion constant
  type(field) :: current, previous    ! Current and previus temperature fields

  real(kind=dp) :: dt     ! Time step
  integer :: nsteps       ! Number of time steps
  integer, parameter :: image_interval = 10 ! Image output interval

  integer :: rows, cols  ! Field dimensions

  character(len=85) :: input_file, arg  ! Input file name and command line arguments

  integer :: iter

  logical :: using_input_file

  ! Default values for grid size and time steps
  rows = 200
  cols = 200
  nsteps = 500
  using_input_file = .false.

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
     call read_input(previous, input_file)
     call copy_fields(previous, current)
  else
     call initialize_field_metadata(previous, rows, cols)
     call initialize_field_metadata(current, rows, cols)
     call initialize(previous)
     call initialize(current)
  end if

  ! Draw the picture of the initial state
  call output(current, 0)

  ! Largest stable time step
  dt = current%dx2 * current%dy2 / &
       & (2.0 * a * (current%dx2 + current%dy2))

  ! Main iteration loop, save a picture every
  ! image_interval steps
  do iter = 1, nsteps
     call evolve(current, previous, a, dt)
     if (mod(iter, image_interval) == 0) then
        call output(current, iter)
     end if
     call swap_fields(current, previous)
  end do

  call finalize(current)
  call finalize(previous)

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

end program
