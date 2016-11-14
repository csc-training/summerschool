program exchange
  use mpi
  implicit none
  integer, parameter :: size = 100
  integer :: rc, myid, ntasks, count
  integer :: status(MPI_STATUS_SIZE)
  integer :: message(size)
  integer :: receiveBuffer(size)

  call mpi_init(rc)
  call mpi_comm_rank(MPI_COMM_WORLD, myid, rc)
  call mpi_comm_size(MPI_COMM_WORLD, ntasks, rc)

  message = myid

  ! TODO: Implement sending and receiving as defined in the assignment
  if ( myid == 0 ) then

     write(*,'(A10,I3,A10,I3)') 'Rank: ', myid, &
          ' received ', receiveBuffer(1)
  else if (myid == 1) then

     write(*,'(A10,I3,A10,I3)') 'Rank: ', myid, &
          ' received ', receiveBuffer(1)
  end if

  call mpi_finalize(rc)

end program exchange
