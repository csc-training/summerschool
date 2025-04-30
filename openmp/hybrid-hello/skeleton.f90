program hello
    use omp_lib
    use mpi
    implicit none
    integer :: my_id, tid, rc
    integer :: provided, required=MPI_THREAD_FUNNELED

    ! TODO: Initialize MPI with thread support.

    ! TODO: Find out the MPI rank and thread ID of each thread and print
    !       out the results.

    ! TODO: Investigate the provided thread support level.

    call MPI_Finalize(rc)
end program hello
