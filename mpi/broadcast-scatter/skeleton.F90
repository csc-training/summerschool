program bcast_scatter
  use mpi_f08
  use iso_fortran_env, only : REAL64
  implicit none

  integer, parameter :: size=12
  integer :: ntasks, rank, i
  integer, dimension(size) :: buf
  integer, dimension(size**2) :: printbuf
  real(REAL64) :: t0, t1

  call mpi_init()
  call mpi_comm_size(MPI_COMM_WORLD, ntasks)
  call mpi_comm_rank(MPI_COMM_WORLD, rank)

  ! Initialize buffer
  call init_buffer(buf)

  ! Print data that will be sent
  call print_buffer(buf)

  ! Start timing
  call mpi_barrier(MPI_COMM_WORLD);
  t0 = mpi_wtime()

  ! Send everywhere
  ! TODO: Implement the broadcast of the array buf

  ! End timing
  call mpi_barrier(MPI_COMM_WORLD);
  t1 = mpi_wtime()

  ! Print data that was received
  call print_buffer(buf)
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
