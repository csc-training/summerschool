program parallel_io_example
    use mpi_f08
    implicit none


    ! How many integers to write, total from all MPI processes
    integer, parameter :: numElements = 32

    ! Enables or disables debug printing of file contents. Printing is not practical for large files,
    ! so we enable/disable this based on 'numElements'.
    logical, parameter :: doDebugPrint = (numElements <= 100)

    integer :: ierr, rank, ntasks, numElementsPerRank
    ! Repeat time measurements this many times
    integer, parameter :: repeatCount = 5
    integer :: i
    character(len=32) :: filename
    ! Storage for the data on each rank
    integer, allocatable :: localData(:)
    ! for time measurements
    real :: startTime, endTime

    call mpi_init(ierr)
    call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)
    call mpi_comm_size(MPI_COMM_WORLD, ntasks, ierr)

    if (mod(numElements, ntasks) /= 0) then
        if (rank == 0) then
            print *, 'numElements must be divisible by the number of MPI tasks!'
        end if
        call mpi_abort(MPI_COMM_WORLD, 1, ierr)
    end if

    numElementsPerRank = numElements / ntasks

    ! Create data array, each element initialized to value 'rank'
    allocate(localData(numElementsPerRank))
    localData = rank

    ! Print some statistics
    if (rank == 0) then
        print *, 'Writing total of ', numElements, ' integers, ', numElementsPerRank, ' from each rank.'
        print *, 'Total bytes to write: ', numElements * 4, ' (', (numElements * 4) / 1024.0 / 1024.0, ' MB)'
    end if

    ! ########## "spokesperson write"
    filename = 'single_writer.dat'

    do i = 1, repeatCount
        startTime = mpi_wtime()
        call single_writer(localData, filename)
        endTime = mpi_wtime()

        if (rank == 0) then
            print *, 'i = ', i, ' : Time taken for single_writer: ', endTime - startTime, ' seconds'
        end if
    end do

    if (rank == 0 .and. doDebugPrint) then
        print *, trim(filename), ' file contents:'
        call debug_read_file(filename)
    end if

    ! ########## Collective write
    filename = 'collective_write.dat'

    do i = 1, repeatCount
        startTime = mpi_wtime()
        call collective_write(localData, filename)
        endTime = mpi_wtime()

        if (rank == 0) then
            print *, 'i = ', i, ' : Time taken for collective_write: ', endTime - startTime, ' seconds'
        end if
    end do

    if (rank == 0 .and. doDebugPrint) then
        print *, trim(filename), ' file contents:'
        call debug_read_file(filename)
    end if

    call MPI_Finalize(ierr)

contains

    subroutine single_writer(localData, filename)
        integer, intent(in) :: localData(:)
        character(len=*), intent(in) :: filename
        integer :: ierr, rank, ntasks
        integer, allocatable :: recvBuffer(:)
        integer :: numElementsPerRank, totalNumElements
        integer :: fh

        ! Get MPI rank and world size
        call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)
        call mpi_comm_size(MPI_COMM_WORLD, ntasks, ierr)

        ! We assume that each rank has the same amount of data
        numElementsPerRank = size(localData)
        totalNumElements = ntasks * numElementsPerRank

        ! "Spokesperson strategy": Send all data to rank == 0 and write it from there.
        ! Rank 0 has to allocate a receive/gather buffer to hold the full data.
        ! Note that the receive buffer for MPI_Gather can be unallocated in other processes.
        ! This saves memory and is OK for MPI
        if (rank == 0) allocate(recvBuffer(totalNumElements))

        ! Gather data to rank 0, each rank sends 'numElementsPerRank' integers.
        ! Note that MPI_Gather automatically orders the data by sending rank
        call mpi_gather(localData, numElementsPerRank, MPI_INTEGER, &
                        recvBuffer, numElementsPerRank, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)

        ! Standard Fortran file write from rank 0
        if (rank == 0) then
            open(newunit=fh, file=filename, form='unformatted', access='stream', action='write', status='replace')
            write(fh) recvBuffer
            close(fh)
        end if
    end subroutine

    subroutine collective_write(localData, filename)
        integer, intent(in) :: localData(:)
        character(len=*), intent(in) :: filename
        integer :: ierr, rank, ntasks, numElementsPerRank
        type(MPI_File) :: file
        integer(kind=MPI_OFFSET_KIND) :: offset, fileSize

        ! We assume that each rank has the same amount of data
        numElementsPerRank = size(localData)

        call mpi_file_open(MPI_COMM_WORLD, filename, &
            MPI_MODE_CREATE + MPI_MODE_WRONLY, MPI_INFO_NULL, file, ierr)

        ! MPI_File_open does NOT truncate the file if it already exists, so we do it manually using MPI_File_set_size().
        ! We could just truncate to zero size and let MPI handle resizing during writing,
        ! but in this case we know what the file size should be => can truncate directly to the final size.

        ! Compute the file size in bytes for truncation. For this we need the number of MPI processes
        call mpi_comm_size(MPI_COMM_WORLD, ntasks, ierr)
        fileSize = ntasks * numElementsPerRank * 4_MPI_OFFSET_KIND

        call mpi_file_set_size(file, fileSize, ierr);

        ! Use the MPI rank of this process to calculate write offset
        call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)

        ! Offset is always given in bytes
        offset = rank * numElementsPerRank * 4_MPI_OFFSET_KIND

        call mpi_file_write_at_all(file, offset, localData, numElementsPerRank, MPI_INTEGER, MPI_STATUS_IGNORE, ierr)
        call mpi_file_close(file, ierr)
    end subroutine

    ! Debugging helper, prints out file contents. You don't have to modify this
    subroutine debug_read_file(filename)
        character(len=*), intent(in) :: filename
        integer :: ierr, rank, fh, fileSize, numInts
        logical :: exists
        integer, allocatable :: readInts(:)

        call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)

        if (rank == 0) then
            inquire(file=filename, exist=exists)
            if (exists) then
                open(newunit=fh, file=filename, form='unformatted', access='stream', action='read', status='old')
                inquire(unit=fh, size=fileSize)

                numInts = fileSize / 4
                allocate(readInts(numInts))
                read(fh) readInts
                close(fh)

                write(*, '( *(I0) )') readInts

            else
                print *, 'Failed to open file ', trim(filename)
            end if
        end if
    end subroutine

end program