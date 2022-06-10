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
  MPI_Datatype particletype, temptype, types[cnt];
  MPI_Aint disp[cnt], dist[2], lb, extent;
  int blocklen[cnt];
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
      strcpy(particles[i].label, "H");
    }
  }
  /* define the datatype for the struct particle */
  types[0]=MPI_FLOAT;
  types[1]=MPI_INT;
  types[2]=MPI_CHAR;
  blocklen[0]=3;
  blocklen[1]=1;
  blocklen[2]=2;
  MPI_Get_address(&particles[0].coords, &disp[0]);
  MPI_Get_address(&particles[0].charge, &disp[1]);
  MPI_Get_address(&particles[0].label, &disp[2]);
  disp[2] -= disp[0];
  disp[1] -= disp[0];
  disp[0] = 0;
  MPI_Type_create_struct(cnt, blocklen, disp, types, &particletype);
  MPI_Type_commit(&particletype);

  /* check extent (not really necessary on most platforms) */
  MPI_Type_get_extent(particletype, &lb, &extent);
  MPI_Get_address(&particles[0], &dist[0]);
  MPI_Get_address(&particles[1], &dist[1]);
  if (extent != (dist[1]-dist[0])) {
    temptype = particletype;
    lb = 0;
    extent = disp[1] - disp[0];
    MPI_Type_create_resized(temptype, lb, extent, &particletype);
    MPI_Type_commit(&particletype);
    MPI_Type_free(&temptype);
  }

  /* communicate using the created particletype */
  t1 = MPI_Wtime();
  if (myid == 0) {
    for (i=0; i < reps; i++)
      MPI_Send(particles, n, particletype, 1, i, MPI_COMM_WORLD);
  } else if (myid == 1) {
    for (i=0; i < reps; i++)
      MPI_Recv(particles, n, particletype, 0, i, MPI_COMM_WORLD,
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
