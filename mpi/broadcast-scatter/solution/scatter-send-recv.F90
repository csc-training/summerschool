program scatter
  use mpi_f08
  use iso_fortran_env, only : REAL64
  implicit none

  integer, parameter :: size=12
  integer :: ntasks, rank, i, block_size
  integer, dimension(size) :: sendbuf, recvbuf
  integer, dimension(size**2) :: printbuf
  real(REAL64) :: t0, t1

  recvbuf(:) = -1

  call mpi_init()
  call mpi_comm_size(MPI_COMM_WORLD, ntasks)
  call mpi_comm_rank(MPI_COMM_WORLD, rank)

  ! Initialize buffer
  call init_buffer(sendbuf)

  ! Print data that will be sent
  call print_buffer(sendbuf)

  ! Start timing
  call mpi_barrier(MPI_COMM_WORLD);
  t0 = mpi_wtime()

  ! Send everywhere
  if (mod(size, ntasks) /= 0) then
     if (rank == 0) then
        print *, "Size not divisible by the number of tasks. This program will fail."
     end if
     call mpi_abort(MPI_COMM_WORLD, -1)
  end if

  block_size = size/ntasks
  if(rank == 0) then
     do i=1, ntasks-1
        call mpi_send(sendbuf(i*block_size + 1:), block_size, MPI_INTEGER, i, 123, MPI_COMM_WORLD)
     enddo

     ! Scatter also the local part
     recvbuf(:block_size) = sendbuf(:block_size)
  else
     call mpi_recv(recvbuf, block_size, MPI_INTEGER, 0, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE)
  endif

  ! End timing
  call mpi_barrier(MPI_COMM_WORLD);
  t1 = mpi_wtime()

  ! Print data that was received
  call print_buffer(recvbuf)
  if (rank == 0) then
      write(*, *) 'Time elapsed: ', t1 - t0, 's'
  endif

  call mpi_finalize()

contains

  subroutine init_buffer(buffer)
    implicit none
    integer, dimension(:), intent(out) :: buffer
    integer, parameter :: bufsize = size
    integer :: i
    if(rank==0) then
      do i = 1, size
         buffer(i) = i
      end do
    else
     do i=1, size
         buffer(i) = -1
     enddo
    endif
  end subroutine init_buffer


  subroutine print_buffer(buffer)
    implicit none
    integer, dimension(:), intent(in) :: buffer
    integer, parameter :: bufsize = size
    integer :: i
    character(len=40) :: pformat

    write(pformat,'(A,I3,A)') '(A4,I2,":",', bufsize, 'I3)'

    call mpi_gather(buffer, bufsize, MPI_INTEGER, &
         & printbuf, bufsize, MPI_INTEGER, &
         & 0, MPI_COMM_WORLD)

    if (rank == 0) then
       do i = 1, ntasks
          write(*,pformat) 'Task', i - 1, printbuf((i-1)*bufsize+1:i*bufsize)
       end do
       print *
    end if
  end subroutine print_buffer

end program
