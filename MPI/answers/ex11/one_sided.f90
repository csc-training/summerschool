program basic
  implicit none
  include 'mpif.h'
  integer, parameter :: size = 100
  integer :: rc, myid, ntasks
  integer :: message(size)
  integer :: receiveBuffer(size)
  
  integer(kind=MPI_ADDRESS_KIND) :: winsize, disp
  integer :: window

  call mpi_init(rc)
  call mpi_comm_rank(MPI_COMM_WORLD, myid, rc)
  call mpi_comm_size(MPI_COMM_WORLD, ntasks, rc)

  message = myid

  winsize = sizeof(receiveBuffer) * size
  ! Create window corresponding to the receive buffer
  call mpi_win_create(receiveBuffer, winsize, sizeof(receiveBuffer), &
                      MPI_INFO_NULL, MPI_COMM_WORLD, window, rc)

  ! Send and receive as defined in the assignment
  call mpi_win_fence(0, window, rc)
  if ( myid < ntasks-1 ) then
     disp = 0
     call mpi_put(message, size, MPI_INTEGER, myid + 1, disp, size, &
                  MPI_INTEGER, window, rc)
     write(*,'(A,I0,A,I0,A,I0)') 'Origin: ', myid, &
          ' Put elements: ',size, &
          '. Target: ', myid+1
  end if
  call mpi_win_fence(0, window, rc)

  if ( myid > 0 ) then
     write(*,'(A,I0,A,I0)') 'Target: ', myid, &
          ' First element: ', receiveBuffer(1)
  end if

  call mpi_win_free(window, rc)
  call mpi_finalize(rc)

end program basic
