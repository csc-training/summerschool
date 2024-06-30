#include <cstdio>
#include <mpi.h>
#include <omp.h>

int main(int argc, char *argv[])
{
    int my_id, thread_id_tag, omp_rank, ntasks, prev_rank, next_rank, rbuf, nrecv, msg;
    int provided;
    int required = MPI_THREAD_MULTIPLE;
    MPI_Status status;

    /* Initialize MPI with thread support. */
    MPI_Init_thread(&argc, &argv, required, &provided);

    // Get the number of MPI tasks/processes
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    /* Find out the MPI rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &omp_rank);

    // Set prev_rank and next_rank ranks. Treat boundaries with MPI_PROC_NULL
    if (omp_rank == 0){
        prev_rank = MPI_PROC_NULL;
        next_rank = omp_rank + 1;
    }
    else if (omp_rank == ntasks - 1){
        prev_rank = omp_rank - 1;
        next_rank = MPI_PROC_NULL;
    }
    else{
        next_rank = omp_rank + 1;
        prev_rank = omp_rank - 1;
    }

    # pragma omp parallel shared(omp_rank) private(my_id, msg, thread_id_tag)  // Some variables need to be private to avoid other threads overwriting them.
    {
    my_id = omp_get_thread_num();
    msg = my_id;
    thread_id_tag = 1000000 + my_id;  // Make a unique tag.
    
    //if(omp_rank == 1){
        // Current rank sends a thread id using MPI to the same thread in rank+1
        MPI_Send(&msg, 1, MPI_INT, next_rank, thread_id_tag, MPI_COMM_WORLD);
        printf("Rank %i thread %i sent %i to rank %i\n", omp_rank, my_id, msg, next_rank);

        // Thread i in current rank receives msg from thread i in prev_rank
        MPI_Recv(&msg, 1, MPI_INT, prev_rank, thread_id_tag, MPI_COMM_WORLD, &status);
        printf("Rank %i thread %i received %i from rank %i\n", omp_rank, my_id, my_id, prev_rank);

        MPI_Get_count(&status, MPI_INT, &nrecv);
    //}

    
    } // end # pragma omp parallel

    /* TODO: Investigate the provided thread support level. */
    int got_required = provided == required;
    printf("Required thread support level provided (0=no, 1=yes): %d. Thread support level: %d\n", got_required, provided); 

    MPI_Finalize();
    return 0;
}
