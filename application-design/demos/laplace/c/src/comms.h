#if defined(HAVE_MPI)
  #include "mpi.h"
#endif

#if defined(HAVE_CUDA)
  #include "devices_cuda.h"
#elif defined(HAVE_HIP)
  #include "devices_hip.h"
#elif defined(HAVE_KOKKOS)
  #include "devices_kokkos.h"
#else
  #include "devices_host.h"
#endif

int comms_get_procs();
int comms_get_rank();
int comms_get_node_procs();
int comms_get_node_rank();
void comms_barrier_procs();
void comms_finalize_procs();
void comms_init_procs(int *argc, char **argv[]);
void comms_reduce_procs(double *sbuf, int count);
