program scatter
  use mpi
  implicit none

  integer, parameter :: size=24
  integer :: ntasks, myid, ierr, i, color, sub_comm
  integer, dimension(size) :: message, recvbuf
  integer, dimension(size**2) :: printbuf
  integer, allocatable :: tmp

  call mpi_init(ierr)
  call mpi_comm_size(MPI_COMM_WORLD, ntasks, ierr)
  call mpi_comm_rank(MPI_COMM_WORLD, myid, ierr)

  call init_buffers
  recvbuf=-1
  call print_buffers(message)

  if (size<ntasks) then
     if (myid == 0) then
        print *, "Size is too small. Increase size or decrease number of tasks to have at least one element sent"
     end if
     call mpi_abort(MPI_COMM_WORLD, -1, ierr)
  end if

  allocate(tmp(size/ntasks))
  if(myid == 0) then
  recvbuf(1:(size/ntasks))=message(1:(size/ntasks))
  do i=1, ntasks-1
     tmp(1:(size/ntasks))=message(i*(size/ntasks)+1:(i+1)*(size/ntasks))
     call mpi_send(tmp, size/ntasks, MPI_INTEGER, i,i,MPI_COMM_WORLD, ierr)
  enddo
  else
  call mpi_recv(tmp, size/ntasks, MPI_INTEGER, 0, myid, MPI_COMM_WORLD, status,ierr)
  recvbuf(1:(size/ntasks))=tmp(1:(size/ntasks))
  endif

  call print_buffers(recvbuf)
  call mpi_finalize(ierr)

contains

  subroutine init_buffers
    implicit none
    integer :: i
    if(myid==0) then
      do i = 1, size
         message(i) = i
      end do
    else
     do i=1, size
         message(i)= -1
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

    if (myid == 0) then
       do i = 1, ntasks
          write(*,pformat) 'Task', i - 1, printbuf((i-1)*bufsize+1:i*bufsize)
       end do
       print *
    end if
  end subroutine print_buffers

end program
