program exchange
  use mpi_f08
  implicit none
  integer, parameter :: arraysize = 100000, msgsize = 100
  integer :: rc, rank, ntasks
  integer :: message(arraysize)
  integer :: receiveBuffer(arraysize)

  call mpi_init(rc)
  call mpi_comm_rank(MPI_COMM_WORLD, rank, rc)
  call mpi_comm_size(MPI_COMM_WORLD, ntasks, rc)

  message = rank
  receiveBuffer = -1


  ! Order of message passing is as follows:
  ! 1. rank 0 sends message to rank 1
  ! 2. rank 1 receives message from rank 0
  ! 3. rank 1 sends message to rank 0
  ! 4. rank 0 receives message from rank 1

  if (rank == 0) then
     call mpi_send(message, msgsize, MPI_INTEGER, 1, &
          1, MPI_COMM_WORLD, rc)
     call mpi_recv(receiveBuffer, arraysize, MPI_INTEGER, 1,  &
          2, MPI_COMM_WORLD, MPI_STATUS_IGNORE, rc)
     write(*,'(A10,I3,A10,I3, A17, I3)') 'Rank: ', rank, &
          ' received ', msgsize, ' elements, first ', receiveBuffer(1)
  else if (rank == 1) then
     call mpi_recv(receiveBuffer, arraysize, MPI_INTEGER, 0,  &
          1, MPI_COMM_WORLD, MPI_STATUS_IGNORE, rc)
     call mpi_send(message, msgsize, MPI_INTEGER, 0, &
          2, MPI_COMM_WORLD, rc)
     write(*,'(A10,I3,A10,I3, A17, I3)') 'Rank: ', rank, &
          ' received ', msgsize, ' elements, first ', receiveBuffer(1)
  end if

  call mpi_finalize(rc)

end program exchange
