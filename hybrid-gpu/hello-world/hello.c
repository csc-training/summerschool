#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENACC
#include <openacc.h>
#endif
#include<mpi.h>

int main(int argc, char *argv[])
{
    int i, myid, ntasks;
#ifdef _OPENACC
    acc_device_t devtype;
#endif

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (myid == 0) {
        printf("Number of MPI tasks: %d\n", ntasks);
    }

    printf("[%d] Hello world!\n", myid);
#ifdef _OPENACC
    devtype = acc_get_device_type();
    printf("[%d] Number of available OpenACC devices: %d\n",
            myid, acc_get_num_devices(devtype));
    printf("[%d] Type of available OpenACC devices: %d\n", myid, devtype);
#else
    printf("[%d] Code compiled without OpenACC\n", myid);
#endif

    MPI_Finalize();
    return 0;
}
