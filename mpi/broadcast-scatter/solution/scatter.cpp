#include <mpi.h>
#include <cstdio>
#include <vector>

void print_buffers(int *printbuffer, int *sendbuffer, int buffersize);
void init_buffers(int *sendbuffer, int *recvbuffer, int buffersize);


int main(int argc, char *argv[])
{
    int ntasks, myid, color,size=12;
    std::vector<int> sendbuf(size);
    std::vector<int> recvbuf(size);
    std::vector<int> printbuf(size*size);
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);


    /* Initialize message buffers */
    init_buffers(sendbuf.data(), recvbuf.data(), size);

    /* Print data that will be sent */
    print_buffers(printbuf.data(), sendbuf.data(), size);

    /* Send  everywhere */
    if (ntasks > size) {
        if (myid == 0) {
            fprintf(stderr, "Size too small to scatter at least one element.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    std::vector<int> tmp(size/ntasks);
    if( myid ==0){
      for(int i=0; i<ntasks; i++){
          for (int j=0; j<size/ntasks; j++)
          {
             tmp[j]=i*(size/ntasks)+j;
             if(i==0){
             recvbuf[j]=tmp[j];
           }
          }
          MPI_Send(tmp.data(), size/ntasks, MPI_INT, i, i, MPI_COMM_WORLD);
          }
     }
     else
     {
       MPI_Recv(tmp.data(), size/ntasks, MPI_INT, 0, myid, MPI_COMM_WORLD, &status);
       for (int j=0;j<size/ntasks;j++)
       {
          recvbuf[j]=tmp[j];
       }
     }
    /* Print data that was received */
    print_buffers(printbuf.data(), recvbuf.data(), size);

    MPI_Finalize();
    return 0;
}


void init_buffers(int *sendbuffer, int *recvbuffer, int buffersize)
{
    int rank, i;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0)
    {
    for (i = 0; i < buffersize; i++) {
        recvbuffer[i] = -1;
        sendbuffer[i] = i;
    }
    }
    else
    {
     for (i = 0; i < buffersize; i++) {
        recvbuffer[i] = -1;
        sendbuffer[i] = -1;
    }
    }
}


void print_buffers(int *printbuffer, int *sendbuffer, int buffersize)
{
    int i, j, rank, ntasks;

    MPI_Gather(sendbuffer, buffersize, MPI_INT,
               printbuffer, buffersize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    if (rank == 0) {
        for (j = 0; j < ntasks; j++) {
            printf("Task %i:", j);
            for (i = 0; i < buffersize; i++) {
                printf(" %2i", printbuffer[i + buffersize * j]);
            }
            printf("\n");
         }
    }
}
