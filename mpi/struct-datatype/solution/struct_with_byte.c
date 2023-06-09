#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>


int main(int argc, char *argv[])
{
  int n=1000, cnt=3, reps=10000;
  typedef struct{
    float coords[3];
    int charge;
    char label[2];
  } particle;
  particle particles[n];
  MPI_Aint lb0, lb1, extent;
  int i, j, myid, ntasks;
  double t1, t2;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  /* fill in some values for the particles */
  if (myid == 0) {
    for (i=0; i < n; i++) {
      for (j=0; j < 3; j++)
        particles[i].coords[j] = (float)rand()/(float)RAND_MAX*10.0;
      particles[i].charge = 54;
      strcpy(particles[i].label, "Xe");
    }
  }
  /* determine the true extent of one particle struct */
  MPI_Get_address(&particles[0], &lb0);
  MPI_Get_address(&particles[1], &lb1);
  extent = lb1 - lb0;

  /* send and receive using the MPI_BYTE datatype */
  t1 = MPI_Wtime();
  if (myid == 0) {
    for (i=0; i < reps; i++)
      MPI_Send(particles, n*extent, MPI_BYTE, 1, i, MPI_COMM_WORLD);
  } else if (myid == 1) {
    for (i=0; i < reps; i++)
      MPI_Recv(particles, n*extent, MPI_BYTE, 0, i, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
  }
  t2 = MPI_Wtime();

  printf("Time: %i, %e \n", myid, (t2-t1)/(double)reps);
  printf("Check: %i: %s %f %f %f \n", myid, particles[n-1].label,
          particles[n-1].coords[0], particles[n-1].coords[1],
          particles[n-1].coords[2]);

  MPI_Finalize();
  return 0;
}
