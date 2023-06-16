program datatype_struct
  use mpi_f08
  use iso_fortran_env, only : REAL64
  implicit none

  integer, parameter :: n = 1000, reps=10000

  type particle
     real :: coords(3)
     integer :: charge
     character(len=2) :: label
  end type particle

  type(particle) :: particles(n)

  integer :: i, ierror, myid

  type(mpi_datatype) :: particle_mpi_type, temp_type
  type(mpi_datatype) :: types(3)
  integer :: blocklen(3)
  integer(kind=MPI_ADDRESS_KIND) :: disp(3)
  integer(kind=MPI_ADDRESS_KIND) :: lb, extent

  real(REAL64) :: t1, t2

  call mpi_init(ierror)
  call mpi_comm_rank(MPI_COMM_WORLD, myid, ierror)

  ! Fill in some values for the particles
  if(myid == 0) then
    do i = 1, n
      call random_number(particles(i)%coords)
      particles(i)%charge = 54
      particles(i)%label = 'Xe'
    end do
  end if

  ! Define datatype for the struct
  types = (/ MPI_REAL, MPI_INTEGER, MPI_CHARACTER  /)
  blocklen = (/ 3, 1, 2 /)
  call MPI_GET_ADDRESS(particles(1)%coords, disp(1), ierror)
  call mpi_get_address(particles(1)%charge, disp(2), ierror)
  call mpi_get_address(particles(1)%label, disp(3), ierror)
  do i = 3, 1, -1
    disp(i) = disp(i) - disp(1)
  end do

  call mpi_type_create_struct(3, blocklen, &
      disp, types, particle_mpi_type, ierror)
  call mpi_type_commit(particle_mpi_type, ierror)

  ! Check extent
  call mpi_type_get_extent(particle_mpi_type, lb, extent, ierror)
  call mpi_get_address(particles(1), disp(1), ierror)
  call mpi_get_address(particles(2), disp(2), ierror)
  if (extent /= disp(2) - disp(1)) then
    temp_type = particle_mpi_type
    lb = 0
    extent = disp(2) - disp(1)
    call mpi_type_create_resized(temp_type, lb, extent, particle_mpi_type, ierror)
    call mpi_type_commit(particle_mpi_type, ierror)
    call mpi_type_free(temp_type, ierror)
  end if

  ! Communicate using the created particletype
  ! Multiple sends are done for better timing
  t1 = mpi_wtime()
  if(myid == 0) then
     do i = 1, reps
        call mpi_send(particles, n, particle_mpi_type, 1, i, &
                      MPI_COMM_WORLD, ierror)
     end do
  else if(myid == 1) then
     do i = 1, reps
        call mpi_recv(particles, n, particle_mpi_type, 0, i, &
                      MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror)
     end do
  end if
  t2=mpi_wtime()

  write(*,*) "Time: ", myid, (t2-t1) / reps
  write(*,*) "Check:", myid, particles(n)%label, particles(n)%coords(1), &
                       particles(n)%coords(2), particles(n)%coords(3)

  ! Free datatype
  call mpi_type_free(particle_mpi_type, ierror)

  call mpi_finalize(ierror)

end program datatype_struct
