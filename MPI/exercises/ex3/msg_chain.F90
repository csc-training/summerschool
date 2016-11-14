program basic
  use mpi
  implicit none
  integer, parameter :: size = 100
  integer :: rc, myid, ntasks, count
  integer :: status(MPI_STATUS_SIZE)
  integer :: message(size)
  integer :: receiveBuffer(size)

  call mpi_init(rc)
  call mpi_comm_rank(MPI_COMM_WORLD, myid, rc)
  call mpi_comm_sizeMPI_COMM_WORLD, ntasks, rc)

  message = myid

  ! TODO: Send and receive as defined in the assignment
  if ( myid < ntasks-1 ) then

     write(*,'(A10,I3,A20,I8,A,I3,A,I3)') 'Sender: ', myid, &
          ' Sent elements: ',size, &
          '. Tag: ', myid+1, '. Receiver: ', myid+1
  end if

  if ( myid > 0 ) then

     write(*,'(A10,I3,A,I3)') 'Receiver: ', myid, &
          ' First element: ', receiveBuffer(1)
  end if

  call mpi_finalize(rc)

end program basic
