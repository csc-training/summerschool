program basic
  use mpi
  use iso_fortran_env, only : REAL64

  implicit none
  integer, parameter :: size = 10000000
  integer :: rc, myid, ntasks
  integer :: message(size)
  integer :: receiveBuffer(size)
  integer :: status(MPI_STATUS_SIZE)

  real(REAL64) :: t0, t1

  call mpi_init(rc)
  call mpi_comm_rank(MPI_COMM_WORLD, myid, rc)
  call mpi_comm_size(MPI_COMM_WORLD, ntasks, rc)

  message = myid

  ! Start measuring the time spent in communication
  call mpi_barrier(mpi_comm_world, rc)
  t0 = mpi_wtime()

  if ( (myid > 0) .and. (myid < ntasks-1) ) then
     ! Send and receive messages
     call mpi_sendrecv(message, size, MPI_INTEGER, myid + 1, myid + 1, &
          receiveBuffer, size, MPI_INTEGER, myid - 1, MPI_ANY_TAG, &
          MPI_COMM_WORLD, status, rc)
     write(*,'(A10,I3,A20,I8,A,I3,A,I3)') 'Sender: ', myid, &
          ' Sent elements: ', size, &
          '. Tag: ', myid + 1, '. Receiver: ', myid + 1
  else if (myid < ntasks-1) then
     ! Only send a message
     call mpi_send(message, size, MPI_INTEGER, myid+1, &
          myid+1, MPI_COMM_WORLD, rc)
     write(*,'(A10,I3,A20,I8,A,I3,A,I3)') 'Sender: ', myid, &
          ' Sent elements: ', size, &
          '. Tag: ', myid + 1, '. Receiver: ', myid + 1
  else if (myid > 0) then
     ! Only receive a message
     call mpi_recv(receiveBuffer, size, MPI_INTEGER, myid - 1,  &
          myid, MPI_COMM_WORLD, status, rc)
     write(*,'(A10,I3,A,I3)') 'Receiver: ', myid, &
          ' First element: ', receiveBuffer(1)
  end if

  ! Finalize measuring the time and print it out
  t1 = mpi_wtime()
  call mpi_barrier(mpi_comm_world, rc)
  call flush(6)

  write(*, '(A20, I3, A, F6.3)') 'Time elapsed in rank', myid, ':', t1-t0

  call mpi_finalize(rc)

end program basic
