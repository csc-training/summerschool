program chain
  use mpi
  implicit none
  integer, parameter :: size = 100
  integer :: rc, myid, ntasks
  integer :: message(size)
  integer :: receiveBuffer(size)
  integer :: status(MPI_STATUS_SIZE,2)

  integer :: source, destination
  integer :: count
  integer :: requests(2)

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

  ! Receive messages in the back ground
  call mpi_irecv(receiveBuffer, size, MPI_INTEGER, source, &
       MPI_ANY_TAG, MPI_COMM_WORLD, requests(1), rc)
  ! Send messages in the back ground
  call mpi_isend(message, size, MPI_INTEGER, destination, &
       myid + 1, MPI_COMM_WORLD, requests(2), rc)

  ! Blocking wait for messages
  call mpi_waitall(2, requests, status, rc)

  ! Use status parameter to find out the no. of elements received
  call mpi_get_count(status(:,1), MPI_INTEGER, count, rc)
  write(*,'(A10,I3,A20,I8,A,I3,A,I3)') 'Sender: ', myid, &
       ' Sent elements: ', size, &
       '. Tag: ', myid + 1, &
       '. Receiver: ', destination
  write(*,'(A10,I3,A20,I8,A,I3,A,I3)') 'Receiver: ', myid, &
       'received elements: ', count, &
       '. Tag: ', status(MPI_TAG, 1), &
       '. Sender:   ', status(MPI_SOURCE, 1)

  call mpi_finalize(rc)
end program chain
