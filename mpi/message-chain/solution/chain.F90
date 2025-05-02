program basic
  use mpi_f08
  use iso_fortran_env, only : REAL64

  implicit none
  integer, parameter :: size = 10000000
  integer :: rc, rank, ntasks, sendTag, recvTag
  integer :: message(size)
  integer :: receiveBuffer(size)

  real(REAL64) :: t0, t1

  integer :: source, destination

  call mpi_init(rc)
  call mpi_comm_rank(MPI_COMM_WORLD, rank, rc)
  call mpi_comm_size(MPI_COMM_WORLD, ntasks, rc)

  message = rank
  receiveBuffer = -1

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

  ! Note that the send tag must be valid (>= 0) even if our destination is MPI_PROC_NULL
  sendTag = rank + 1
  recvTag = rank

  if (rank == 0) then
      write(*,*) '#### Case A: Chain with MPI_Send and MPI_Recv ####'
  endif

  ! Start measuring the time spent in communication
  call mpi_barrier(mpi_comm_world, rc)
  t0 = mpi_wtime()

  ! Send and receive messages
  call mpi_send(message, size, MPI_INTEGER, destination, sendTag, MPI_COMM_WORLD, rc)
  call mpi_recv(receiveBuffer, size, MPI_INTEGER, source, recvTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE, rc)
  write(*,'(A10,I3,A20,I8,A,I3,A,I3)') 'Sender: ', rank, &
          ' Sent elements: ', size, &
          ' Tag: ', sendTag, ' Receiver: ', destination
  write(*,'(A10,I3,A,I3,A,I3)') 'Receiver: ', rank, &
          ' Tag: ', recvTag, ' First element: ', receiveBuffer(1)

  ! Finalize measuring the time and print it out
  t1 = mpi_wtime()
  call mpi_barrier(mpi_comm_world, rc)
  call flush(6)

  call print_ordered(t1 - t0)

  if (rank == 0) then
    write(*,*) '#### End case A ####'
  endif

  if (rank == 0) then
    write(*,*) '#### Case B: Chain with MPI_Sendrecv ####'
  endif

  call mpi_barrier(mpi_comm_world, rc)
  t0 = mpi_wtime()

  call mpi_sendrecv(message, size, MPI_INTEGER, destination, sendTag, &
       receiveBuffer, size, MPI_INTEGER, source, recvTag, &
       MPI_COMM_WORLD, MPI_STATUS_IGNORE, rc)

  write(*,'(A10,I3,A20,I8,A,I3,A,I3)') 'Sender: ', rank, &
    ' Sent elements: ', size, &
    ' Tag: ', sendTag, ' Receiver: ', destination
  write(*,'(A10,I3,A,I3,A,I3)') 'Receiver: ', rank, &
    ' Tag: ', recvTag, ' First element: ', receiveBuffer(1)

  t1 = mpi_wtime()
  call mpi_barrier(mpi_comm_world, rc)
  call flush(6)

  call print_ordered(t1 - t0)

  ! This mpi_sendrecv implementation of the message chain is more efficient
  ! compared to separately doing mpi_send and mpi_recv.
  ! This is because the other version does mpi_send (blocking) first from all ranks
  ! except for the last one, so only the last process can proceed to the receive stage.
  ! The messages are then received in reverse order, causing the chain to slowly unwind.
  ! The first sender (rank 0) has to wait until all other receives have been processed.
  ! In contrast, the mpi_sendrecv version allows rank 0 to receive and continue as soon as
  ! rank 1 has finished its send, and so on.

  if (rank == 0) then
    write(*,*) '#### End case B ####'
  endif

  if (rank == 0) then
    write(*,*) '#### Case C: Chain with alternating send and receive ####'
  endif

  call mpi_barrier(mpi_comm_world, rc)
  t0 = mpi_wtime()

  if (modulo(rank, 2) == 0) then
    call mpi_send(message, size, MPI_INTEGER, destination, sendTag, MPI_COMM_WORLD, rc)
    call mpi_recv(receiveBuffer, size, MPI_INTEGER, source, recvTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE, rc)
  else
    call mpi_recv(receiveBuffer, size, MPI_INTEGER, source, recvTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE, rc)
    call mpi_send(message, size, MPI_INTEGER, destination, sendTag, MPI_COMM_WORLD, rc)
  endif

  write(*,'(A10,I3,A20,I8,A,I3,A,I3)') 'Sender: ', rank, &
    ' Sent elements: ', size, &
    ' Tag: ', sendTag, ' Receiver: ', destination
  write(*,'(A10,I3,A,I3,A,I3)') 'Receiver: ', rank, &
    ' Tag: ', recvTag, ' First element: ', receiveBuffer(1)

  t1 = mpi_wtime()
  call mpi_barrier(mpi_comm_world, rc)
  call flush(6)

  call print_ordered(t1 - t0)

  if (rank == 0) then
  write(*,*) '#### End case C ####'
  endif



  call mpi_finalize(rc)

contains

  subroutine print_ordered(t)
    implicit none
    real(REAL64) :: t

    integer i

    if (rank == 0) then
       write(*, '(A20, I3, A, F6.3)') 'Time elapsed in rank', rank, ':', t
       do i=1, ntasks-1
           call mpi_recv(t, 1, MPI_DOUBLE_PRECISION, i, 11,  &
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE, rc)
           write(*, '(A20, I3, A, F6.3)') 'Time elapsed in rank', i, ':', t
       end do
    else
       call mpi_send(t, 1, MPI_DOUBLE_PRECISION, 0, 11,  &
                         MPI_COMM_WORLD, rc)
    end if
  end subroutine print_ordered

end program basic
