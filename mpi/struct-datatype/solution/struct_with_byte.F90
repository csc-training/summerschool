program datatype_struct
  use mpi_f08
  use iso_fortran_env, only : REAL64
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
  type(mpi_datatype):: types(cnt)
  integer :: blocklen(cnt)
  integer(kind=MPI_ADDRESS_KIND) :: disp(cnt)
  integer(kind=MPI_ADDRESS_KIND) :: lb1, lb2, extent
  integer :: nbytes

  real(REAL64) :: t1, t2

  call mpi_init(ierror)
  call mpi_comm_rank(MPI_COMM_WORLD, myid, ierror)
  call mpi_comm_size(MPI_COMM_WORLD, ntasks, ierror)

  ! insert some data for the particle struct
  if (myid == 0) then
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

  t1 = mpi_wtime()
  ! send and receive using the MPI_BYTE datatype
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
