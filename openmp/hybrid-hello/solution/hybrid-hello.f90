program hello
    use omp_lib
    use mpi
    implicit none
    integer :: my_id, tid, rc
    integer :: provided, required=MPI_THREAD_FUNNELED

    call MPI_Init_thread(required, provided, rc)
    !call MPI_Init(rc)
    !call MPI_Query_thread(provided, rc)

    call MPI_Comm_rank(MPI_COMM_WORLD, my_id, rc)

    !$omp parallel private(tid)
    tid = omp_get_thread_num()
    write(*,'(A,I2,A,I2)') "I'm thread ", tid, ' in process ', my_id
    !$omp end parallel

    if (my_id == 0) then
        write(*,*) ''
        write(*,'(A,I1)') 'Provided thread support level: ', provided
        write(*,'(A,I1,A)') '  ', MPI_THREAD_SINGLE, &
                            & ' - MPI_THREAD_SINGLE'
        write(*,'(A,I1,A)') '  ', MPI_THREAD_FUNNELED, &
                            & ' - MPI_THREAD_FUNNELED'
        write(*,'(A,I1,A)') '  ', MPI_THREAD_SERIALIZED, &
                            & ' - MPI_THREAD_SERIALIZED'
        write(*,'(A,I1,A)') '  ', MPI_THREAD_MULTIPLE, &
                            & ' - MPI_THREAD_MULTIPLE'
    end if
    call MPI_Finalize(rc)
end program hello
