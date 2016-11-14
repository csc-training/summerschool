program coll_exer
  use mpi
  implicit none

  integer, parameter :: n_mpi_tasks = 4

  integer :: ntasks, rank, ierr, i, color, sub_comm
  integer, dimension(2*n_mpi_tasks) :: sendbuf, recvbuf
  integer, dimension(2*n_mpi_tasks**2) :: printbuf
  integer, dimension(n_mpi_tasks) :: offsets, counts

  call mpi_init(ierr)
  call mpi_comm_size(MPI_COMM_WORLD, ntasks, ierr)
  call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)

  if (ntasks /= n_mpi_tasks) then
     if (rank == 0) then
        print *, "Run this program with ", n_mpi_tasks, " tasks."
     end if
     call mpi_abort(MPI_COMM_WORLD, -1, ierr)
  end if

  ! First collective operation to send (0,1,..,7) everywhere
  ! Initialize sendbuf and broadcast
  call init_buffers
  call mpi_bcast(sendbuf, 2*n_mpi_tasks, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
  call print_buffers(sendbuf)

  ! Initialize buffers for a)
  call init_buffers
  call print_buffers(sendbuf)

  ! Scatter the elements from task 0
  call mpi_scatter(sendbuf, 2, MPI_INTEGER, recvbuf, 2, MPI_INTEGER, &
       & 0, MPI_COMM_WORLD, ierr)
  call print_buffers(recvbuf)

  ! Gather varying size data to task 1
  call init_buffers
  counts = (/1,1,2,4/)
  offsets(1) = 0
  do i = 2, ntasks
     offsets(i) = offsets(i-1) + counts(i-1)
  end do
  call mpi_gatherv(sendbuf, counts(rank+1), MPI_INTEGER, recvbuf, counts, &
       & offsets, MPI_INTEGER, 1, MPI_COMM_WORLD, ierr)
  call print_buffers(recvbuf)

  ! Create new communicator and reduce the data
  call init_buffers
  if (rank / 2 == 0) then
     color = 1
  else
     color = 2
  end if
  call mpi_comm_split(MPI_COMM_WORLD, color, rank, sub_comm, ierr)
  call mpi_reduce(sendbuf, recvbuf, 2*n_mpi_tasks, MPI_INTEGER, MPI_SUM, 0, &
       & sub_comm, ierr)

  call print_buffers(recvbuf)
  call mpi_barrier(MPI_COMM_WORLD, ierr)
  call flush(6)
  call mpi_finalize(ierr)

contains

  subroutine init_buffers
    implicit none
    integer :: i

    do i = 1, 2*n_mpi_tasks
       recvbuf(i) = -1
       sendbuf(i) = i + 2*n_mpi_tasks * rank - 1
    end do
  end subroutine init_buffers


  subroutine print_buffers(buffer)
    implicit none
    integer, dimension(:), intent(in) :: buffer
    integer, parameter :: bufsize = 2*n_mpi_tasks
    integer :: i
    character(len=40) :: pformat

    write(pformat,'(A,I3,A)') '(A4,I2,":",', bufsize, 'I3)'

    call mpi_gather(buffer, bufsize, MPI_INTEGER, &
         & printbuf, bufsize, MPI_INTEGER, &
         & 0, MPI_COMM_WORLD, ierr)

    if (rank == 0) then
       do i = 1, ntasks
          write(*,pformat) 'Task', i - 1, printbuf((i-1)*bufsize+1:i*bufsize)
       end do
       print *
    end if
  end subroutine print_buffers

end program coll_exer
