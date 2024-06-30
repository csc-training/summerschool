#include <cstdio>
#include <mpi.h>
#include <omp.h>

int main(int argc, char *argv[])
{
    int my_id, thread_id_tag1, thread_id_tag2, ntasks, omp_rank, send_rank, recv_rank1, recv_rank2, nrecv, msg;
    int provided;
    int required = MPI_THREAD_MULTIPLE;
    MPI_Status status;

    /* Initialize MPI with thread support. */
    MPI_Init_thread(&argc, &argv, required, &provided);

    /* TODO: Investigate the provided thread support level. */
    //int got_required = provided == required;
    //printf("Required thread support level provided (0=no, 1=yes): %d. Thread support level: %d\n", got_required, provided);
    if (provided != required) {
        printf("MPI does not support MPI_THREAD_MULTIPLE.\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
        return 0;
    }

    // Get the number of MPI tasks/processes
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    if (ntasks < 3) {
        printf("Use at least 3 MPI tasks.\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
        return 0;
    }

    /* Find out the MPI rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &omp_rank);

    send_rank = 0;
    recv_rank1 = 1;
    recv_rank2 = 2;

    # pragma omp parallel shared(omp_rank, send_rank, recv_rank1, recv_rank2) private(my_id, msg, thread_id_tag1, thread_id_tag2, nrecv)  // Some variables need to be private to avoid other threads overwriting them.
    {
    my_id = omp_get_thread_num();
    msg = my_id;
    thread_id_tag1 = 1000000 + my_id;  // Make a unique tag.
    thread_id_tag2 = 2000000 + my_id;  // Make a unique tag.
    
    if(omp_rank == send_rank){
        // Current rank sends a thread id using MPI to the same thread in recv_rank1
        MPI_Send(&msg, 1, MPI_INT, recv_rank1, thread_id_tag1, MPI_COMM_WORLD);
        // Current rank sends a thread id using MPI to the same thread in recv_rank2
        MPI_Send(&msg, 1, MPI_INT, recv_rank2, thread_id_tag2, MPI_COMM_WORLD);
        //printf("Rank %i thread %i sent %i to rank %i\n", omp_rank, my_id, msg, next_rank);
    }
    else if (omp_rank == recv_rank1){
        // Thread i in omp_rank receives msg from thread i in send_rank
        MPI_Recv(&msg, 1, MPI_INT, send_rank, thread_id_tag1, MPI_COMM_WORLD, &status);
        printf("Rank %i thread %i received %i from rank %i\n", omp_rank, my_id, my_id, send_rank);
        //MPI_Get_count(&status, MPI_INT, &nrecv);
    }
    else if (omp_rank == recv_rank2){
        // Thread i in omp_rank receives msg from thread i in send_rank
        MPI_Recv(&msg, 1, MPI_INT, send_rank, thread_id_tag2, MPI_COMM_WORLD, &status);
        printf("Rank %i thread %i received %i from rank %i\n", omp_rank, my_id, my_id, send_rank);
        //MPI_Get_count(&status, MPI_INT, &nrecv);
    }

    
    } // end # pragma omp parallel 

    MPI_Finalize();
    return 0;
}
