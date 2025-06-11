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

    call single_writer(localData, filename)

    if (rank == 0 .and. doDebugPrint) then
        print *, trim(filename), ' file contents:'
        call debug_read_file(filename)
    end if

    ! ########## Collective write
    filename = 'collective_write.dat'

    call collective_write(localData, filename)

    if (rank == 0 .and. doDebugPrint) then
        print *, trim(filename), ' file contents:'
        call debug_read_file(filename)
    end if

    call MPI_Finalize(ierr)

contains

    subroutine single_writer(localData, filename)
        integer, intent(in) :: localData(:)
        character(len=*), intent(in) :: filename

        ! Gets called from all MPI ranks. 'localData' contains different data on each rank.
        ! TODO: Gather contents of 'localData' to one MPI process and write it all to file 'filename' ("spokesperson" strategy).
        ! The output should be ordered such that data from rank 0 comes first, then rank 1, and so on

        ! You can assume that 'localData' has same length in all MPI processes

    end subroutine

    subroutine collective_write(localData, filename)
        integer, intent(in) :: localData(:)
        character(len=*), intent(in) :: filename

        ! TODO: Like single_writer(), but implement a parallel write using MPI_File_write_at_all()

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