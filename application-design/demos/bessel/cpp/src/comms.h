#if defined(HAVE_MPI)
  #include "mpi.h"
#endif

#if defined(HAVE_CUDA)
  #include "devices_cuda.h"
#elif defined(HAVE_HIP)
  #include "devices_hip.h"
#else
  #include "devices_host.h"
#endif

namespace comms{
  int get_procs();
  int get_rank();
  int get_node_procs();
  int get_node_rank();

  void barrier_procs();
  void reduce_procs(float *sbuf, int count);
  
  void init_procs(int *argc, char **argv[]);
  void finalize_procs();
  
}
