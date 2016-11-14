program basic
  use mpi
  implicit none
  integer, parameter :: size = 100
  integer :: rc, myid, ntasks, count
  integer :: status(MPI_STATUS_SIZE)
  integer :: message(size)
  integer :: receiveBuffer(size)
  integer :: source, destination

  call mpi_init(rc)
  call mpi_comm_rank(MPI_COMM_WORLD, myid, rc)
  call mpi_comm_size(MPI_COMM_WORLD, ntasks, rc)

  message = myid

  if (myid < ntasks-1) then
     destination = myid + 1
  else
     destination = MPI_PROC_NULL
  end if

  ! Send and receive as defined in exercises
  if (myid > 0) then
     source = myid - 1
  else
     source = MPI_PROC_NULL
  end if

  call mpi_sendrecv(message, size, MPI_INTEGER, destination, myid + 1, &
       receiveBuffer, size, MPI_INTEGER,source, MPI_ANY_TAG, &
       MPI_COMM_WORLD, status, rc)

  write(*,'(A10,I3,A20,I8,A,I3,A,I3)') 'Sender: ', myid, &
       ' Sent elements: ', size, &
       '. Tag: ', myid + 1, '. Receiver: ', destination

  call mpi_finalize(rc)
end program basic
