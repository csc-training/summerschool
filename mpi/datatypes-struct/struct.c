#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>


int main(int argc, char *argv[])
{
  int n=1000, reps=10000;

  typedef struct {
    float coords[3];
    int charge;
    char label[2];
  } particle;

  particle particles[n];

  int i, j, myid;
  double t1, t2;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  // Fill in some values for the particles
  if (myid == 0) {
    for (i=0; i < n; i++) {
      for (j=0; j < 3; j++) {
        particles[i].coords[j] = (float)rand()/(float)RAND_MAX*10.0;
      }
      particles[i].charge = 54;
      strcpy(particles[i].label, "Xe");
    }
  }

  // Define datatype for the struct
  // TODO

  // Check extent
  // TODO

  // Communicate using the created particletype
  // Multiple sends are done for better timing
  t1 = MPI_Wtime();
  if (myid == 0) {
    for (i=0; i < reps; i++) {
      // TODO: send
    }
  } else if (myid == 1) {
    for (i=0; i < reps; i++) {
      // TODO: receive
    }
  }
  t2 = MPI_Wtime();

  printf("Time: %i, %e \n", myid, (t2-t1)/(double)reps);
  printf("Check: %i: %s %f %f %f \n", myid, particles[n-1].label,
          particles[n-1].coords[0], particles[n-1].coords[1],
          particles[n-1].coords[2]);

  // Free datatype
  // TODO

  MPI_Finalize();
  return 0;
}
