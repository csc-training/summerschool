program scatter
  use mpi_f08
  implicit none

  integer, parameter :: size=12
  integer :: ntasks, rank, ierr, i
  integer :: block_size
  integer, dimension(size) :: sendbuf, recvbuf
  integer, dimension(size**2) :: printbuf
  type(mpi_status) :: status

  call mpi_init(ierr)
  call mpi_comm_size(MPI_COMM_WORLD, ntasks, ierr)
  call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)

  ! Initialize buffers
  call init_buffers

  ! Print data that will be sent
  call print_buffers(sendbuf)

  ! Send everywhere
  if (mod(size, ntasks) /= 0) then
     if (rank == 0) then
        print *, "Size not divisible by the number of tasks. This program will fail."
     end if
     call mpi_abort(MPI_COMM_WORLD, -1, ierr)
  end if

  block_size = size/ntasks
  if(rank == 0) then
     do i=1, ntasks-1
        call mpi_send(sendbuf(i*block_size + 1:), block_size, MPI_INTEGER, i, i, MPI_COMM_WORLD, ierr)
     enddo

     ! Scatter also the local part
     recvbuf(:block_size) = sendbuf(:block_size)
  else
     call mpi_recv(recvbuf, block_size, MPI_INTEGER, 0, rank, MPI_COMM_WORLD, status, ierr)
  endif

  ! Print data that was received
  call print_buffers(recvbuf)

  call mpi_finalize(ierr)

contains

  subroutine init_buffers
    implicit none
    integer :: i
    if(rank==0) then
      do i = 1, size
         recvbuf(i) = -1
         sendbuf(i) = i
      end do
    else
     do i=1, size
         recvbuf(i) = -1
         sendbuf(i) = -1
     enddo
    endif
  end subroutine init_buffers


  subroutine print_buffers(buffer)
    implicit none
    integer, dimension(:), intent(in) :: buffer
    integer, parameter :: bufsize = size
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

end program
