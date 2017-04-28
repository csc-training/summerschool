program basic
  use mpi
  implicit none
  integer, parameter :: size = 100
  integer :: rc, myid, ntasks
  integer :: message(size)
  integer :: receiveBuffer(size)
  integer :: status(MPI_STATUS_SIZE)

  integer :: source, destination

  call mpi_init(rc)
  call mpi_comm_rank(MPI_COMM_WORLD, myid, rc)
  call mpi_comm_size(MPI_COMM_WORLD, ntasks, rc)

  message = myid

  ! Set source and destination ranks
  if (myid < ntasks-1) then
     destination = myid + 1
  else
     destination = MPI_PROC_NULL
  end if
  if (myid > 0) then
     source = myid - 1
  else
     source = MPI_PROC_NULL
  end if

  ! Send messages
  call mpi_send(message, size, MPI_INTEGER, destination, &
       myid + 1, MPI_COMM_WORLD, rc)
  write(*,'(A10,I3,A20,I8,A,I3,A,I3)') 'Sender: ', myid, &
       ' Sent elements: ', size, &
       '. Tag: ', myid + 1, '. Receiver: ', destination
  ! Receive messages
  call mpi_recv(receiveBuffer, size, MPI_INTEGER, source,  &
       MPI_ANY_TAG, MPI_COMM_WORLD, status, rc)
  write(*,'(A10,I3,A,I3)') 'Receiver: ', myid, &
       ' First element: ', receiveBuffer(1)

  call mpi_finalize(rc)
end program basic
