program basic
  use mpi_f08
  use iso_fortran_env, only : REAL64

  implicit none
  integer, parameter :: size = 10000000
  integer :: rc, rank, ntasks
  integer :: message(size)
  integer :: receiveBuffer(size)
  type(mpi_status) :: status
  integer :: requests(2)

  integer :: ndims, dims(1), cart_id
  type(mpi_comm) :: cart_comm
  logical :: reorder, periods(1)

  real(REAL64) :: t0, t1

  integer :: source, destination

  call mpi_init(rc)
  call mpi_comm_rank(MPI_COMM_WORLD, rank, rc)
  call mpi_comm_size(MPI_COMM_WORLD, ntasks, rc)

  message = rank
  receiveBuffer = -1

  ndims = 1
  dims(1) = ntasks
  periods(1) = .true.
  reorder = .true.
  call mpi_cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, cart_comm, rc)
  call mpi_cart_shift(cart_comm, 0, 1, source, destination, rc)
  call mpi_comm_rank(cart_comm, cart_id, rc)


  ! Start measuring the time spent in communication
  call mpi_barrier(mpi_comm_world, rc)
  t0 = mpi_wtime()

  ! Send and receive messages
  call mpi_sendrecv(message, size, MPI_INTEGER, destination, destination, &
                    receiveBuffer, size, MPI_INTEGER, source, cart_id, &
                    cart_comm, status, rc)
  write(*,'(A10,I3,A20,I8,A,I3,A,I3)') 'Sender: ', rank, &
       ' Sent elements: ', size, &
       '. Tag: ', rank + 1, &
       '. Receiver: ', destination
  write(*,'(A10,I3,A,I3)') 'Receiver: ', rank, &
          ' First element: ', receiveBuffer(1)

  ! Finalize measuring the time and print it out
  t1 = mpi_wtime()
  call mpi_barrier(mpi_comm_world, rc)
  call flush(6)

  call print_ordered(t1 - t0)
  call mpi_comm_free(cart_comm, rc)

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
