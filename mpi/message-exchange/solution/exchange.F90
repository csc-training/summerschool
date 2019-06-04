program exchange
  use mpi_f08
  implicit none
  integer, parameter :: msgsize = 100
  integer :: rc, myid, ntasks
  type(mpi_status) :: status
  integer :: message(msgsize)
  integer :: receiveBuffer(msgsize)

  call mpi_init(rc)
  call mpi_comm_rank(MPI_COMM_WORLD, myid, rc)
  call mpi_comm_size(MPI_COMM_WORLD, ntasks, rc)

  message = myid

  ! Send and receive as defined in the assignment
  if (myid == 0) then
     call mpi_send(message, msgsize, MPI_INTEGER, 1, &
          1, MPI_COMM_WORLD, rc)
     call mpi_recv(receiveBuffer, msgsize, MPI_INTEGER, 1,  &
          2, MPI_COMM_WORLD, status, rc)
     write(*,'(A10,I3,A10,I3)') 'Rank: ', myid, &
          ' received ', receiveBuffer(1)
  else if (myid == 1) then
     call mpi_send(message, msgsize, MPI_INTEGER, 0, &
          2, MPI_COMM_WORLD, rc)
     call mpi_recv(receiveBuffer, msgsize, MPI_INTEGER, 0,  &
          1, MPI_COMM_WORLD, status, rc)
     write(*,'(A10,I3,A10,I3)') 'Rank: ', myid, &
          ' received ', receiveBuffer(1)
  end if

  call mpi_finalize(rc)

end program exchange
