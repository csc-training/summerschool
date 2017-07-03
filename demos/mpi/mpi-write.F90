program output
      use mpi
      implicit none
      integer :: err, i, myid, file, intsize
      integer :: status(MPI_STATUS_SIZE)
      integer, parameter :: count=100
      integer(kind=mpi_offset_kind) :: disp
      integer :: buf(count)

      call mpi_init(err)
      call mpi_comm_rank(MPI_COMM_WORLD, myid, err)
      do i = 1, count
        buf(i) = myid * count + i
      end do

      call mpi_file_open(MPI_COMM_WORLD, 'test', &
          MPI_MODE_CREATE + MPI_MODE_WRONLY, MPI_INFO_NULL, file, err)
      intsize = sizeof(count)
      disp = myid * count * intsize
      call mpi_file_write_at(file, disp, buf, count, MPI_INTEGER, status, err)
      call mpi_file_close(file, err)

      call mpi_finalize(err)
end program output
