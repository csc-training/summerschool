#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <mpi.h>

#define DATASIZE   64
#define WRITER_ID   0

void single_writer(int, int *, int);
void many_writers(int, int *, int);


int main(int argc, char *argv[])
{
    int my_id, ntasks, i, localsize;
    int *localvector;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

    if (ntasks > 64) {
        fprintf(stderr, "Datasize (64) should be divisible by number "
                "of tasks.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (DATASIZE % ntasks != 0) {
        fprintf(stderr, "Datasize (64) should be divisible by number "
                "of tasks.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    localsize = DATASIZE / ntasks;
    localvector = (int *) malloc(localsize * sizeof(int));

    for (i = 0; i < localsize; i++) {
        localvector[i] = i + 1 + localsize * my_id;
    }

    many_writers(my_id, localvector, localsize);

    free(localvector);

    MPI_Finalize();
    return 0;
}

void single_writer(int my_id, int *localvector, int localsize)
{
    FILE *fp;
    int *fullvector;

    fullvector = (int *) malloc(DATASIZE * sizeof(int));

    MPI_Gather(localvector, localsize, MPI_INT, fullvector, localsize,
               MPI_INT, WRITER_ID, MPI_COMM_WORLD);

    if (my_id == WRITER_ID) {
        if ((fp = fopen("singlewriter.dat", "wb")) == NULL) {
            fprintf(stderr, "Error: %d (%s)\n", errno, strerror(errno));
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        } else {
            fwrite(fullvector, sizeof(int), DATASIZE, fp);
            fclose(fp);
            printf("Wrote %d elements to file singlewriter.dat\n", DATASIZE);
        }
    }

    free(fullvector);
}

void many_writers(int my_id, int *localvector, int localsize)
{
    FILE *fp;
    char filename[64];

    sprintf(filename, "manywriters-%d.dat", my_id);

    if ((fp = fopen(filename, "wb")) == NULL) {
        fprintf(stderr, "Error: %d (%s)\n", errno, strerror(errno));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    } else {
        fwrite(localvector, sizeof(int), localsize, fp);
        fclose(fp);
        printf("Wrote %d elements to file manywriters-%d.dat\n", localsize,
                my_id);
    }
}

