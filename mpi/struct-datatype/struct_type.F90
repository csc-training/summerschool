program datatype_struct
  use mpi_f08
  use use iso_fortran_env, only : REAL64
  implicit none

  integer, parameter :: n = 1000, cnt=3, reps=10000

  type particle
     real :: coords(3)
     integer :: charge
     character(len=2) :: label
  end type particle

  type(particle) :: particles(n)

  integer :: i, ierror,  myid,  ntasks, tag

  type(mpi_datatype) :: particle_mpi_type, temp_type
  integer :: types(cnt), blocklen(cnt)
  integer(kind=MPI_ADDRESS_KIND) :: disp(cnt)
  integer(kind=MPI_ADDRESS_KIND) :: lb, extent

  real(REAL64) :: t1,t2

  call mpi_init(ierror)
  call mpi_comm_rank(MPI_COMM_WORLD, myid, ierror)
  call mpi_comm_size(MPI_COMM_WORLD, ntasks, ierror)

  ! insert some data for the particle struct
  if(myid == 0) then
    do i = 1, n
      call random_number(particles(i)%coords)
      particles(i)%charge = 54
      particles(i)%label = 'Xe'
    end do
  end if

  ! TODO: define the datatype for type particle

  ! TODO: Check extent.
  ! (Not really neccessary on most systems.)

  ! communicate using the created particletype
  t1 = mpi_wtime()
  if(myid == 0) then
     do i = 1, reps  ! multiple sends for better timing
       ! TODO: send
     end do
  else if(myid == 1) then
     do i = 1, reps
       ! TODO: receive
     end do
  end if
  t2=mpi_wtime()

  ! TODOs end

  write(*,*) "Time: ", myid, (t2-t1) / reps
  write(*,*) "Check:", myid, particles(n)%label, particles(n)%coords(1), &
                       particles(n)%coords(2), particles(n)%coords(3)

  call mpi_type_free(particle_mpi_type, ierror)
  call mpi_finalize(ierror)

end program datatype_struct
