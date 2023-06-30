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
    call h5open_f(errc)
    call h5pcreate_f(H5P_FILE_ACCESS_F, property_list, errc)
    call h5pset_fapl_mpio_f(property_list, MPI_COMM_WORLD, MPI_INFO_NULL, errc)
    call h5fcreate_f("data.h5", H5F_ACC_TRUNC_F, file_handle, errc, &
         & access_prp=property_list)
    call h5pclose_f(property_list, errc)

    ! Create the dataset
    dims = datasize
    call h5screate_simple_f(1, dims, file_space, errc)
    call h5dcreate_f(file_handle, 'data', H5T_NATIVE_INTEGER, &
         & file_space, dataset_id, errc)
    call h5sclose_f(file_space, errc)

    ! Select a hyperslab of the file dataspace
    counts = localsize
    offsets = my_id * localsize
    call h5dget_space_f(dataset_id, file_space, errc)
    call h5sselect_hyperslab_f(file_space, H5S_SELECT_SET_F, offsets, &
         & counts, errc)

    ! Now we can write our local data to the correct position in the
    ! dataset. Here we use collective write, but independent writes are
    ! also possible.
    call h5screate_simple_f(1, counts, mem_space, errc)
    call h5pcreate_f(H5P_DATASET_XFER_F, property_list, errc)
    call h5pset_dxpl_mpio_f(property_list, H5FD_MPIO_COLLECTIVE_F, errc)

    call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, writevector, dims, errc, &
         & file_space_id=file_space, mem_space_id=mem_space, &
         & xfer_prp=property_list)

    ! Close all opened HDF5 handles
    call h5pclose_f(property_list, errc)
    call h5sclose_f(mem_space, errc)
    call h5sclose_f(file_space, errc)
    call h5dclose_f(dataset_id, errc)
    call h5fclose_f(file_handle, errc)
    call h5close_f(errc)

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
    call h5open_f(errc)
    call h5pcreate_f(H5P_FILE_ACCESS_F, property_list, errc)
    call h5pset_fapl_mpio_f(property_list, MPI_COMM_WORLD, MPI_INFO_NULL, errc)
    call h5fopen_f("data.h5", H5F_ACC_RDONLY_F, file_handle, errc, &
         & access_prp=property_list)
    call h5pclose_f(property_list, errc)

    ! Open the dataset and get the filespace id
    dims = datasize
    call h5dopen_f(file_handle, 'data', dataset_id, errc)
    call h5dget_space_f(dataset_id, file_space, errc)
    call h5sclose_f(file_space, errc)

    ! Select a hyperslab of the file dataspace
    counts = localsize
    offsets = my_id * localsize
    call h5dget_space_f(dataset_id, file_space, errc)
    call h5sselect_hyperslab_f(file_space, H5S_SELECT_SET_F, offsets, &
         & counts, errc)

    ! Now we can read our local data from the correct position in the
    ! dataset. Here we use collective read, but independent reads are
    ! also possible.
    call h5screate_simple_f(1, counts, mem_space, errc)
    call h5pcreate_f(H5P_DATASET_XFER_F, property_list, errc)
    call h5pset_dxpl_mpio_f(property_list, H5FD_MPIO_COLLECTIVE_F, errc)

    call h5dread_f(dataset_id, H5T_NATIVE_INTEGER, readvector, dims, errc, &
         & file_space_id=file_space, mem_space_id=mem_space, &
         & xfer_prp=property_list)

    ! Close all opened HDF5 handles
    call h5pclose_f(property_list, errc)
    call h5sclose_f(mem_space, errc)
    call h5sclose_f(file_space, errc)
    call h5dclose_f(dataset_id, errc)
    call h5fclose_f(file_handle, errc)
    call h5close_f(errc)

  end subroutine h5_reader

end program hdf5io
