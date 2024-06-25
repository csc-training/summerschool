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

  int i, j, rank;
  double t1, t2;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Fill in some values for the particles
  if (rank == 0) {
    for (i=0; i < n; i++) {
      for (j=0; j < 3; j++) {
        particles[i].coords[j] = (float)rand()/(float)RAND_MAX*10.0;
      }
      particles[i].charge = 54;
      strcpy(particles[i].label, "Xe");
    }
  }

  // Define datatype for the struct
  MPI_Datatype particletype;
  MPI_Datatype types[] = {MPI_FLOAT, MPI_INT, MPI_CHAR};
  int blocklen[] = {3, 1, 2};
  MPI_Aint disp[3];
  MPI_Get_address(&particles[0].coords, &disp[0]);
  MPI_Get_address(&particles[0].charge, &disp[1]);
  MPI_Get_address(&particles[0].label, &disp[2]);
  disp[2] -= disp[0];
  disp[1] -= disp[0];
  disp[0] = 0;
  MPI_Type_create_struct(3, blocklen, disp, types, &particletype);
  MPI_Type_commit(&particletype);

  // Check extent
  MPI_Aint lb, extent, correct_extent, lb0, lb1;
  MPI_Type_get_extent(particletype, &lb, &extent);
  MPI_Get_address(&particles[0], &lb0);
  MPI_Get_address(&particles[1], &lb1);
  correct_extent = lb1 - lb0;
  if (rank == 0) {
    // Extent is usually correct on most platforms by default
    printf("Checking extent:\n");
    printf("  extent:  %ld\n", extent);
    printf("  correct: %ld\n", correct_extent);
    printf("  sizeof:  %ld\n", sizeof(particles[0]));
  }
  if (extent != correct_extent) {
    MPI_Datatype tmp = particletype;
    lb = 0;
    MPI_Type_create_resized(tmp, lb, correct_extent, &particletype);
    MPI_Type_commit(&particletype);
    MPI_Type_free(&tmp);
  }

  // Communicate using the created particletype
  // Multiple sends are done for better timing
  t1 = MPI_Wtime();
  if (rank == 0) {
    for (i=0; i < reps; i++) {
      MPI_Send(particles, n, particletype, 1, i, MPI_COMM_WORLD);
    }
  } else if (rank == 1) {
    for (i=0; i < reps; i++) {
      MPI_Recv(particles, n, particletype, 0, i, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }
  }
  t2 = MPI_Wtime();

  printf("Time: %i, %e \n", rank, (t2-t1)/(double)reps);
  printf("Check: %i: %s %f %f %f \n", rank, particles[n-1].label,
          particles[n-1].coords[0], particles[n-1].coords[1],
          particles[n-1].coords[2]);

  // Free datatype
  MPI_Type_free(&particletype);

  MPI_Finalize();
  return 0;
}
