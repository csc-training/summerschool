program chain
  use mpi_f08
  use iso_fortran_env, only : REAL64

  implicit none
  integer, parameter :: msgsize = 10000000
  integer :: rc, myid, ntasks
  integer :: message(msgsize)
  integer :: receiveBuffer(msgsize)
  type(mpi_status) :: status(2)

  real(REAL64) :: t0, t1

  integer :: source, destination
  integer :: count
  type(mpi_request) :: requests(2)

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

  ! Start measuring the time spent in communication
  call mpi_barrier(mpi_comm_world, rc)
  t0 = mpi_wtime()

  ! Describe the receiving request
  call mpi_recv_init(receiveBuffer, msgsize, MPI_INTEGER, source, &
       MPI_ANY_TAG, MPI_COMM_WORLD, requests(1), rc)
  ! Describe the sending request
  call mpi_send_init(message, msgsize, MPI_INTEGER, destination, &
       myid + 1, MPI_COMM_WORLD, requests(2), rc)

  ! Start communication
  call mpi_startall(2, requests, rc)

  ! Blocking wait for messages
  call mpi_waitall(2, requests, status, rc)
  t1 = mpi_wtime()

  ! Use status parameter to find out the no. of elements received
  call mpi_get_count(status(1), MPI_INTEGER, count, rc)
  write(*,'(A10,I3,A20,I8,A,I3,A,I3)') 'Sender: ', myid, &
       ' Sent elements: ', msgsize, &
       '. Tag: ', myid + 1, &
       '. Receiver: ', destination
  write(*,'(A10,I3,A20,I8,A,I3,A,I3)') 'Receiver: ', myid, &
       'received elements: ', count, &
       '. Tag: ', status(1)%MPI_TAG, &
       '. Sender:   ', status(1)%MPI_SOURCE

  call mpi_barrier(mpi_comm_world, rc)
  call flush(6)

  write(*, '(A20, I3, A, F6.3)') 'Time elapsed in rank', myid, ':', t1-t0

  call mpi_finalize(rc)
end program chain
