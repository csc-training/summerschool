program chain
  use mpi_f08
  use iso_fortran_env, only : REAL64

  implicit none
  integer, parameter :: msgsize = 10000000
  integer :: rc, rank, ntasks
  integer :: message(msgsize)
  integer :: receiveBuffer(msgsize)
  type(mpi_status) :: status(2)

  real(REAL64) :: t0, t1

  integer :: source, destination
  integer :: count
  type(mpi_request) :: requests(2)

  call mpi_init(rc)
  call mpi_comm_rank(MPI_COMM_WORLD, rank, rc)
  call mpi_comm_size(MPI_COMM_WORLD, ntasks, rc)

  message = rank

  ! Set source and destination ranks
  if (rank < ntasks-1) then
     destination = rank + 1
  else
     destination = MPI_PROC_NULL
  end if
  if (rank > 0) then
     source = rank - 1
  else
     source = MPI_PROC_NULL
  end if

  ! Start measuring the time spent in communication
  call mpi_barrier(mpi_comm_world, rc)
  t0 = mpi_wtime()

  ! Receive messages in the back ground
  call mpi_irecv(receiveBuffer, msgsize, MPI_INTEGER, source, &
       MPI_ANY_TAG, MPI_COMM_WORLD, requests(1), rc)
  ! Send messages in the back ground
  call mpi_isend(message, msgsize, MPI_INTEGER, destination, &
       rank + 1, MPI_COMM_WORLD, requests(2), rc)

  ! Blocking wait for messages
  call mpi_waitall(2, requests, status, rc)

  ! Use status parameter to find out the no. of elements received
  call mpi_get_count(status(1), MPI_INTEGER, count, rc)
  write(*,'(A10,I3,A20,I8,A,I3,A,I3)') 'Sender: ', rank, &
       ' Sent elements: ', msgsize, &
       '. Tag: ', rank + 1, &
       '. Receiver: ', destination
  write(*,'(A10,I3,A20,I8,A,I3,A,I3)') 'Receiver: ', rank, &
       'received elements: ', count, &
       '. Tag: ', status(1)%MPI_TAG, &
       '. Sender:   ', status(1)%MPI_SOURCE

  ! Finalize measuring the time and print it out
  t1 = mpi_wtime()
  call mpi_barrier(mpi_comm_world, rc)
  call flush(6)

  write(*, '(A20, I3, A, F6.3)') 'Time elapsed in rank', rank, ':', t1-t0

  call mpi_finalize(rc)
end program chain
