program probe
    use mpi_f08
    implicit none
    integer :: rc, rank, ntasks, i
    integer :: messageLength = 1
    ! Just some tag for the message
    integer :: tag = 123
    integer, allocatable :: message(:), receiveBuffer(:)
    type(mpi_status) :: status

    call mpi_init(rc)
    call mpi_comm_rank(MPI_COMM_WORLD, rank, rc)
    call mpi_comm_size(MPI_COMM_WORLD, ntasks, rc)

    if (ntasks < 2) then
        print *, "Please run with at least 2 MPI processes"
        call mpi_finalize(rc)
        stop
    end if

    if (rank == 1) then

        ! Generate random message size in rank 1 only. Other ranks do not know the size
        messageLength = randomMessageLength()

        allocate(message(messageLength))

        ! fill in a test message: element i has value i
        do i = 1, messageLength
            message(i) = i
        end do

        ! Send the test message to rank 0
        print *, "Rank 1: Sending", messageLength, "integers to rank 0"

        call mpi_send(message, messageLength, MPI_INTEGER, 0, tag, MPI_COMM_WORLD, rc)

    else if (rank == 0) then

        ! Probe for incoming messages from rank 1, without receiving it.
        ! MPI_Probe is blocking, causing the calling rank to wait until an incoming message is detected.
        ! Info about the message is stored into the MPI_Status variable.
        call mpi_probe(1, tag, MPI_COMM_WORLD, status, rc)

        ! Find the number of integers contained in the message that we probed,
        ! and store in 'messageLength'
        call mpi_get_count(status, MPI_INTEGER, messageLength, rc)

        ! Resize the receive buffer so that the full message fits in
        allocate(receiveBuffer(messageLength))

        call mpi_recv(receiveBuffer, messageLength, MPI_INTEGER, 1, &
            tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE, rc)

        print *, "Rank 0: Received", messageLength, "integers from rank 1"
        do i = 1, size(receiveBuffer)
            print *, 'receiveBuffer(', i, ') : ', receiveBuffer(i)
        end do

        ! For Approach 2, we are told that the maximum 'messageLength' is 10.
        ! Therefore the following would work (note the 'count' argument to mpi_recv):

        !allocate(receiveBuffer(10))
        !call mpi_recv(receiveBuffer, 10, MPI_INTEGER, 1, tag, MPI_COMM_WORLD, status, rc)
        !call mpi_get_count(status, MPI_INTEGER, messageLength, rc)

        ! This is OK for MPI because the receive buffer is always large enough to hold the incoming message.
        ! However, this approach potentially wastes memory, and necessitates that we have information about how large the messages can be.
        ! Real programs may benefit from a combination of both approaches:
        !  1. Allocate a reusable receive buffer to some initial size
        !  2. Use MPI_Probe to check that the message can fit; if not, reallocate the buffer
        !  3. Receive and wait for the next message
        ! This approach can minimize the number of mallocs/frees required for repeated message receiving.
        ! Whether it's a good fit for your program depends entirely on the use case.

    end if

    call mpi_finalize(rc)

contains

! Generates a random message length, no need to modify this
function randomMessageLength() result(randomInt)
    implicit none
    real :: randomReal
    integer :: randomInt

    call random_number(randomReal)
    ! range [2, 10]
    randomInt = int(randomReal * 9.0) + 2
end function randomMessageLength


end program probe
