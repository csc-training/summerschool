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

  integer(kind=MPI_ADDRESS_KIND) :: lb1, lb2, extent
  integer :: nbytes

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

  ! Determine the true extent of one particle struct
  call MPI_GET_ADDRESS(particles(1),lb1,ierror)
  call MPI_GET_ADDRESS(particles(2),lb2,ierror)
  extent = lb2 - lb1

  ! Send and receive using the MPI_BYTE datatype
  ! Multiple sends are done for better timing
  t1 = mpi_wtime()
  nbytes = n * extent
  if(myid == 0) then
     do i = 1, reps
        call mpi_send(particles, nbytes, MPI_BYTE, 1, i, &
                      MPI_COMM_WORLD, ierror)
     end do
  else if(myid == 1) then
     do i = 1, reps
        call mpi_recv(particles, nbytes, MPI_BYTE, 0, i, &
                      MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror)
     end do
  end if
  t2=mpi_wtime()

  write(*,*) "Time: ", myid, (t2-t1) / reps
  write(*,*) "Check:", myid, particles(n)%label, particles(n)%coords(1), &
                       particles(n)%coords(2), particles(n)%coords(3)

  call mpi_finalize(ierror)

end program datatype_struct
