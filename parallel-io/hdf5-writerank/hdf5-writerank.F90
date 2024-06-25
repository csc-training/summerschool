program hdf5_writerank
    use mpi
    use hdf5
    implicit none
    ! Declare vars
    integer :: err, myproc, numprocs
    integer(hid_t) :: plist, file, dataspace, dataset, memspace
    integer(hsize_t) :: counts(1)
    ! Initialize MPI
    call mpi_init(err)
    call mpi_comm_rank(MPI_COMM_WORLD, myproc, err)
    call mpi_comm_size(MPI_COMM_WORLD, numprocs, err)

    ! Initialize Fortran HDF5 interface
    call h5open_f(err)
    ! Create a new property list for file access
    call h5pcreate_f(H5P_FILE_ACCESS_F, plist, err)
    ! Store MPI IO communicator info to the file access property list
    call h5pset_fapl_mpio_f(plist, MPI_COMM_WORLD, MPI_INFO_NULL, err)
    ! Create a new HDF5 file named "parallel_out.h5"
    call h5fcreate_f("parallel_out.h5", H5F_ACC_TRUNC_F, file, err, access_prp=plist)
    ! Create a new simple dataspace for the file and open for access
    call h5screate_simple_f(1, int([numprocs], hsize_t), dataspace, err)
    ! Create a new dataset named "MPI_RANKS" for 'file'
    call h5dcreate_f(file, "MPI_RANKS", H5T_NATIVE_INTEGER, dataspace, dataset, err)
    ! Number of blocks to be included in the hyperslab region
    counts(1) = 1
    ! Select a hyperslab region of the file dataspace
    call h5sselect_hyperslab_f(dataspace, H5S_SELECT_SET_F, int([myproc], hsize_t), counts, err)
    ! Create a new simple dataspace for the memory buffer and open for access
    call h5screate_simple_f(1, counts, memspace, err)
    ! Each rank writes its own rank number (partially constructing 'dataset') into a file
    call h5dwrite_f(dataset, H5T_NATIVE_INTEGER, [myproc], int([numprocs], hsize_t), err, memspace, dataspace, H5P_DEFAULT_F)
    
    ! Close all handles
    call h5dclose_f(dataset, err)
    call h5sclose_f(dataspace, err)
    call h5sclose_f(memspace, err)
    call h5fclose_f(file, err)
    call h5pclose_f(plist, err)
    call h5close_f(err)
    call mpi_finalize(err)
end program hdf5_writerank
