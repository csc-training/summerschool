program pario
  use mpi
  use, intrinsic :: iso_fortran_env, only : error_unit, output_unit
  implicit none

  integer, parameter :: datasize = 64, writer_id = 0
  integer :: rc, my_id, ntasks, localsize, i
  integer, dimension(:), allocatable :: localvector

  call mpi_init(rc)
  call mpi_comm_size(mpi_comm_world, ntasks, rc)
  call mpi_comm_rank(mpi_comm_world, my_id, rc)

  if (ntasks > 64) then
    write(error_unit, *) 'Maximum number of tasks is 64!'
    call mpi_abort(MPI_COMM_WORLD, -1, rc)
  end if

  if (mod(datasize, ntasks) /= 0) then
    write(error_unit,*) 'Datasize (64) should be divisible by number of tasks'
    call mpi_abort(MPI_COMM_WORLD, -1, rc)
  end if

  localsize = datasize / ntasks
  allocate(localvector(localsize))

  localvector = [(i + my_id * localsize, i=1,localsize)]

  call mpiio_writer()

  deallocate(localvector)
  call mpi_finalize(rc)

contains

  subroutine mpiio_writer()
    implicit none
    integer :: fh, rc, dsize
    integer(kind=MPI_OFFSET_KIND) :: offset;

    call mpi_type_size(MPI_INTEGER, dsize, rc)
    
    ! TODO: write the output file "mpiio.dat" using MPI IO. Each
    !       rank should write their own local vectors to correct
    !       locations in the output file.

  end subroutine mpiio_writer

end program pario
