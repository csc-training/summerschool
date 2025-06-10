// AdaptiveCpp compilation with cpu & nvidia targets: syclcc -O3 --hipsycl-targets="omp;cuda:sm_80" <code>.cpp 
// AdaptiveCpp compilation with cpu & amd targets:    syclcc -O3 --hipsycl-targets="omp;hip:gfx90a" <code>.cpp 
// OneAPI with cpu & nvidia targets:clang++ -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80  <code>.cpp 
// OneAPI with cpu & amd targets:      icpx -std=c++17 -O3 -fsycl -fsycl-targets=amdgcn-amd-amdhsa,spir64_x86_64 -Xsycl-target-backend=amdgcn-amd-amdhsa  --offload-arch=gfx90a  <code>.cpp 

// Compilation with mpi
// export  MPI_flags=-I/appl/spack/v017/install-tree/gcc-11.2.0/openmpi-4.1.2-bylozw/include -I/appl/spack/v017/install-tree/gcc-11.2.0/openmpi-4.1.2-bylozw/include/openmpi -I/appl/spack/syslibs/include -pthread -L/appl/spack/v017/install-tree/gcc-11.2.0/openmpi-4.1.2-bylozw/lib -L/appl/spack/syslibs/lib -Wl,-rpath,/appl/spack/v017/install-tree/gcc-8.5.0/gcc-11.2.0-zshp2k/lib/gcc/x86_64-pc-linux-gnu/11.2.0 -Wl,-rpath,/appl/spack/v017/install-tree/gcc-8.5.0/gcc-11.2.0-zshp2k/lib64 -Wl,-rpath -Wl,/appl/spack/v017/install-tree/gcc-11.2.0/openmpi-4.1.2-bylozw/lib -Wl,-rpath -Wl,/appl/spack/syslibs/lib -lmpi
// clang++ $MPI_flags    -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 pp_with_usm.cpp 
// This is a port of a CUDA code written for the CUDA training at CSC https://github.com/csc-training/CUDA
// For the compilation check also the instructions for the latest way.
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>
#include <ctime>
#include <chrono>
#include <getopt.h>
#include <mpi.h>

#include <sycl/sycl.hpp>

using namespace sycl;

class add_kernel {
  int* h_usm;
  int N; 

public:
  add_kernel(int* h, int n) : h_usm(h), N(n) {}

  void operator()(nd_item<1> idx) const {
    if(idx.get_global_id(0)<N){
        h_usm[idx.get_global_id(0)]++;
    }
  }
};


void CPUtoCPUtest(int id, int *ha, const int N, double *timer)
{
    double start, stop;
    start=MPI_Wtime();
    // Transfers that uses GPU-aware MPI to transfer data
    if (id == 0) { //Sender process
        MPI_Send(ha, N, MPI_INT, 1, 11, MPI_COMM_WORLD);
        /* Receive the added data back */
        MPI_Recv(ha, N, MPI_INT, 1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else { // Adder process

        MPI_Recv(ha, N, MPI_INT, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         /* Add one*/
        for (int i = 0; i < N; ++i)
        {
            ha[i] += 1.0;
        }
        MPI_Send(ha, N, MPI_INT, 0, 12, MPI_COMM_WORLD);
    }
    stop = MPI_Wtime();
    *timer = stop - start;
}

void GPUtoGPUtestmanual(int id, int *da, int *ha, const int N, const int M, double *timer, queue &q)
{
    double start, stop;
    start=MPI_Wtime();
    // Transfers that uses GPU-aware MPI to transfer data
    if (id == 0) { //Sender process
        /* Send data to rank 1 for addition */
        q.memcpy(ha, da, N * sizeof(int)).wait();
        MPI_Send(ha, N, MPI_INT, 1, 11, MPI_COMM_WORLD);
        /* Receive the added data back */
        MPI_Recv(ha, N, MPI_INT, 1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        q.memcpy(da, ha, N * sizeof(int)).wait();
    } else { // Adder process

        MPI_Recv(ha, N, MPI_INT, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        q.memcpy(da, ha, N * sizeof(int)).wait();
        q.submit([&] (handler& h){
            h.parallel_for(nd_range<1>(static_cast<size_t>(M*((N-1+M)/M)),static_cast<size_t>(M)), add_kernel(da,N));
            });
        q.wait_and_throw();
        q.memcpy(ha, da, N * sizeof(int)).wait();
        MPI_Send(ha, N, MPI_INT, 0, 12, MPI_COMM_WORLD);
    }
    stop = MPI_Wtime();
    *timer = stop - start;
}

void GPUtoGPUtestGPUAware(int id, int *da, const int N, const int M, double *timer, queue &q)
{
    double start, stop;
    start=MPI_Wtime();
    // Transfers that uses GPU-aware MPI to transfer data
    if (id == 0) { //Sender process
        /* Send data to rank 1 for addition */
        MPI_Send(da, N, MPI_INT, 1, 11, MPI_COMM_WORLD);
        /* Receive the added data back */
        MPI_Recv(da, N, MPI_INT, 1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    } else { // Adder process

        MPI_Recv(da, N, MPI_INT, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        q.submit([&] (handler& h){
            h.parallel_for(nd_range<1>(static_cast<size_t>(M*((N-1+M)/M)),static_cast<size_t>(M)), add_kernel(da,N));
            });
        q.wait_and_throw();
        MPI_Send(da, N, MPI_INT, 0, 12, MPI_COMM_WORLD);
    }
    stop = MPI_Wtime();
    *timer = stop - start;
}

void ShowDevice(queue &q) 
{


  // Output platform and device information.
  
  auto device = q.get_device();
  auto p_name = device.get_platform().get_info<info::platform::name>();
  std::cout << "\t\t\t\tPlatform Name: " << p_name << "\n";
  auto p_version = device.get_platform().get_info<info::platform::version>();
  std::cout << "\t\t\t\tPlatform Version: " << p_version << "\n";
  auto d_name = device.get_info<info::device::name>();
  std::cout << "\t\t\t\tDevice Name: " << d_name << "\n";
  auto max_work_group = device.get_info<info::device::max_work_group_size>();
  std::cout << "\t\t\t\tMax Work Group: " << max_work_group << "\n";
  auto max_compute_units = device.get_info<info::device::max_compute_units>();
  std::cout << "\t\t\t\tMax Compute Units: " << max_compute_units << "\n\n";
}


int main(int argc, char *argv[]) 
{
    MPI_Comm intranodecomm;
    int id=0;
    int nprocs=1;
    int noderank=0;
    int nodeprocs=1;
    char machine_name[50];
    int name_len=0;
    int M=256, N;
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) 
    {
        std::cout << "Failed to initialize MPI\n";
        exit(-1);
    }
    
    // Create the communicator, and retrieve the number of MPI ranks.
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    // Determine the rank number.
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    // Get the machine name.
    MPI_Get_processor_name(machine_name, &name_len);
    //Check how many tasks per node by creating local  mpi communicators of the tasks in the same node
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,  MPI_INFO_NULL, &intranodecomm);
    MPI_Comm_rank(intranodecomm, &noderank);
    MPI_Comm_size(intranodecomm, &nodeprocs);

    if (argc < 2) {
        printf("Need the vector length as argument\n");
       N=256*1024*1024;
    }
    else {
         N = atoi(argv[1]);
    }

    //Check how many gpus per node are present
    property_list q_prof{sycl::property::queue::in_order{}};
    auto gpu_devices= device::get_devices(sycl::info::device_type::gpu);
    auto devcount=size( gpu_devices ); 


    /* Due to the test, we need exactly two processes with one GPU for
       each */
    if (nprocs != 2) {
        printf("Need exactly two processes!\n");
        exit(EXIT_FAILURE);
    }
    if (devcount == 0) {
        printf("Could now find any GPU devices.\n");
        exit(EXIT_FAILURE);
    }
    if (nodeprocs > devcount) {
        printf("Not enough GPUs for all processes in the node.\n");
        exit(EXIT_FAILURE);
    }
    // Assign the gpu to each task based on the mpi rank
    queue q{gpu_devices[id],q_prof};
    std::cout << "Rank #" << id << " runs on: " << machine_name  << "\n";
    ShowDevice(q);
    MPI_Barrier(MPI_COMM_WORLD);
    
    double GPUtime,CPUtime;
    std::vector<int> ha(N);
    int* da_usm = malloc_device<int>(N, q);

    // Dummy transfer to remove the overhead of the first communication
    CPUtoCPUtest(id, ha.data(), N, &CPUtime);


    std::fill(ha.begin(), ha.end(), 1);
    // Copy data from host to USM
    q.memcpy(da_usm, ha.data(), N * sizeof(int)).wait();

     /* CPU-to-CPU test */
    CPUtoCPUtest(id, ha.data(), N, &CPUtime);
    if (id  == 0) {
        int errorsum = 0;
        for (int i = 0; i < N; ++i)
            errorsum += ha[i] - 2.0;        
        printf("CPU-CPU time time %lf, errorsum %d\n", CPUtime, errorsum);
    }

    // Dummy transfer to remove the overhead of the first communication
    GPUtoGPUtestGPUAware(id, da_usm, N, M, &GPUtime, q);
    
    // Reinitialize the data
    std::fill(ha.begin(), ha.end(), 1);
    // Copy data from host to USM
    q.memcpy(da_usm, ha.data(), N * sizeof(int)).wait();

    /* GPU-to-GPU test, GPU-aware */
    GPUtoGPUtestGPUAware(id, da_usm, N, M, &GPUtime, q);

    /*Check results, copy device array back to Host*/
    q.memcpy(ha.data(), da_usm, N * sizeof(int)).wait();

    if (id  == 0) {
        int errorsum = 0;
        for (int i = 0; i < N; ++i)
            errorsum += ha[i] - 2.0;        
        printf("GPU-GPU GPU-aware time %lf, errorsum %d\n", GPUtime, errorsum);
    }

    // Dummy transfer to remove the overhead of the first communication
    GPUtoGPUtestmanual(id, da_usm, ha.data() , N, M, &GPUtime, q);

    // Reinitialize the data
    std::fill(ha.begin(), ha.end(), 1);
    // Copy data from host to USM
    q.memcpy(da_usm, ha.data(), N * sizeof(int)).wait();


    /* GPU-to-GPU test, manual transfers */
    GPUtoGPUtestmanual(id, da_usm, ha.data() , N, M, &GPUtime, q);

    /*Check results, copy device array back to Host*/
    q.memcpy(ha.data(), da_usm, N * sizeof(int)).wait();

    if (id  == 0) {
        int errorsum = 0;
        for (int i = 0; i < N; ++i)
            errorsum += ha[i] - 2.0;        
        printf("GPU-GPU manual transfer time %lf, errorsum %d\n", GPUtime, errorsum);
    }

}
