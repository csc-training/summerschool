program basic
  use mpi_f08
  use iso_fortran_env, only : REAL64

  implicit none
  integer, parameter :: size = 10000000
  integer :: rc, rank, ntasks
  integer :: message(size)
  integer :: receiveBuffer(size)
  type(mpi_status) :: status(2)

  real(REAL64) :: t0, t1

  integer :: source, destination
  integer :: count
  type(mpi_request) :: requests(2)

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

  ! Start measuring the time spent in communication
  call mpi_barrier(mpi_comm_world, rc)
  t0 = mpi_wtime()

  ! Initialize communication
  call mpi_recv_init(receiveBuffer, size, MPI_INTEGER, source, rank, &
       MPI_COMM_WORLD, requests(1), rc)
  call mpi_send_init(message, size, MPI_INTEGER, destination, rank + 1, &
       MPI_COMM_WORLD, requests(2), rc)

  ! Start communication
  call mpi_startall(2, requests, rc)

  ! Finalize communication
  call mpi_waitall(2, requests, status, rc)

  ! Use status parameter to find out the no. of elements received
  call mpi_get_count(status(1), MPI_INTEGER, count, rc)
  write(*,'(A10,I3,A20,I8,A,I3,A,I3)') 'Sender: ', rank, &
       ' Sent elements: ', size, &
       '. Tag: ', rank + 1, &
       '. Receiver: ', destination
  write(*,'(A10,I3,A20,I8,A,I3,A,I3)') 'Receiver: ', rank, &
       'received elements: ', count, &
       '. Tag: ', status(1)%mpi_tag, &
       '. Sender:   ', status(1)%mpi_source

  ! Finalize measuring the time and print it out
  t1 = mpi_wtime()
  call mpi_barrier(mpi_comm_world, rc)
  call flush(6)

  call print_ordered(t1 - t0)

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
