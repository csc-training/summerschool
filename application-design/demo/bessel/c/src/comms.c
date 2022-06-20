#include "comms.h"

#if defined(HAVE_MPI)

static int MPI_INITIALIZED = 0;

int comms_get_procs(){
  int comm_size = 1;
  if (MPI_INITIALIZED == 1){
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  }
  return comm_size;
}

int comms_get_rank(){
  int proc_rank = 0;
  if (MPI_INITIALIZED == 1){
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  }
  return proc_rank;
}

int comms_get_node_rank(){
  int node_rank = 0;
  if (MPI_INITIALIZED == 1){
    MPI_Comm node_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);

    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_free(&node_comm);
  }
  return node_rank;
}

int comms_get_node_procs(){
  int node_comm_size = 1;
  if (MPI_INITIALIZED == 1){
    MPI_Comm node_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);

    MPI_Comm_size(node_comm, &node_comm_size);
    MPI_Comm_free(&node_comm);
  }
  return node_comm_size;
}

void comms_barrier_procs(){
  // Synchronize across all MPI processes
  if (MPI_INITIALIZED == 1) 
    MPI_Barrier(MPI_COMM_WORLD);
}

void comms_bcast(int *buf, int count, int root){
  if (MPI_INITIALIZED == 1)
    MPI_Bcast(&buf, count, MPI_INT, root, MPI_COMM_WORLD);
}

void comms_reduce_procs(float *sbuf, int count){
  if (MPI_INITIALIZED == 1){
    float* rbuf;
    if(comms_get_rank() == 0)
      rbuf = (float*)malloc(count * sizeof(float));
    MPI_Reduce(sbuf, rbuf, count, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(comms_get_rank() == 0){
      memcpy(sbuf, rbuf, count * sizeof(float));
      free((void*)rbuf);
    }
  }
}

void comms_init_procs(int *argc, char **argv[]){   
  if(*argc > 1){
    MPI_Init(argc, argv);
    MPI_INITIALIZED = 1;
  }
  // Some device backends require an initialization
  devices_init(comms_get_node_rank());
}

void comms_finalize_procs(){
  // Some device backends also require a finalization
  devices_finalize(comms_get_rank());
  // Finalize MPI if it is used
  if (MPI_INITIALIZED == 1) 
    MPI_Finalize();
}

#else

int comms_get_procs(){
  int comm_size = 1;
  return comm_size;
}

int comms_get_rank(){
  int proc_rank = 0;
  return proc_rank;
}

int comms_get_node_rank(){
  int node_rank = 0;
  return node_rank;
}

int comms_get_node_procs(){
  int node_comm_size = 1;
  return node_comm_size;
}

void comms_barrier_procs(){
}

void comms_bcast(int *buf, int count, int root){
}

void comms_reduce_procs(float *sbuf, int count){
}

void comms_init_procs(int *argc, char **argv[]){
  // Some device backends require an initialization
  devices_init(comms_get_node_rank());
}

void comms_finalize_procs(){
  // Some device backends also require a finalization
  devices_finalize(comms_get_rank());
}

#endif