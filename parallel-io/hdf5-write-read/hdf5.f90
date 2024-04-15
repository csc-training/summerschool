program hdf5io
  use mpi ! HDF5 does not (yet) support mpi f08 interface
  use, intrinsic :: iso_fortran_env, only : error_unit, output_unit
  implicit none

  integer, parameter :: datasize = 64, writer_id = 0
  integer :: rc, my_id, ntasks, localsize, i
  integer, dimension(:), allocatable :: writevector
  integer, dimension(:), allocatable :: readvector

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
  allocate(writevector(localsize))
  allocate(readvector(localsize))
  writevector = [(i + my_id * localsize, i=1,localsize)]

  call h5_writer()
  call h5_reader()

  do i = 1, localsize
    if (readvector(i)/=writevector(i)) then
      print *,"vectors are not equal"
      stop
   end if
  end do

  if (my_id == 0) then 
    print *,"vectors are equal"
  end if

  deallocate(writevector)
  deallocate(readvector)
  call mpi_finalize(rc)

contains

  subroutine h5_writer()
    use hdf5
    implicit none
    integer :: errc
    integer(kind=hid_t) :: property_list, file_handle, file_space, &
         & dataset_id, mem_space
    integer(kind=hsize_t) :: dims(1), counts(1)
    integer(kind=hssize_t) :: offsets(1)

    ! Create the handle for parallel file access property list
    ! and create a new file

    ! Create the dataset

    ! Select a hyperslab of the file dataspace

    ! Now we can write our local data to the correct position in the
    ! dataset. Here we use collective write, but independent writes are
    ! also possible.

    ! Close all opened HDF5 handles

  end subroutine h5_writer


  subroutine h5_reader()
    use hdf5
    implicit none
    integer :: errc
    integer(kind=hid_t) :: property_list, file_handle, file_space, &
         & dataset_id, mem_space
    integer(kind=hsize_t) :: dims(1), counts(1)
    integer(kind=hssize_t) :: offsets(1)

    ! Create the handle for parallel file access property list
    ! and open a file for reading

    ! Open the dataset and get the filespace id

    ! Select a hyperslab of the file dataspace

    ! Now we can read our local data from the correct position in the
    ! dataset. Here we use collective read, but independent reads are
    ! also possible.

    ! Close all opened HDF5 handles

  end subroutine h5_reader

end program hdf5io
