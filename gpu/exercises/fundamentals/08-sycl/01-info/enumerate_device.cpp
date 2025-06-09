// AdaptiveCpp compilation with cpu & nvidia targets: syclcc -O3 --hipsycl-targets="omp;cuda:sm_80" <code>.cpp 
// AdaptiveCpp compilation with cpu & amd targets:    syclcc -O3 --hipsycl-targets="omp;hip:gfx90a" <code>.cpp 
// OneAPI with cpu & nvidia targets:clang++ -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80  <code>.cpp 
// OneAPI with cpu & amd targets:      icpx -std=c++17 -O3 -fsycl -fsycl-targets=amdgcn-amd-amdhsa,spir64_x86_64 -Xsycl-target-backend=amdgcn-amd-amdhsa  --offload-arch=gfx90a  <code>.cpp 

#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>
#include <ctime>
#include <chrono>
#include <getopt.h>

#include <sycl/sycl.hpp>

using namespace sycl;

void ShowDevice(queue &q) {

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
     sycl::property_list q_prof{property::queue::enable_profiling{}, sycl::property::queue::in_order{}};
    std::cout << "\nList Devices\n" << std::endl;

    std::cout << "\tChecking for CPUs\n" << std::endl;
    auto cpu_devices= sycl::device::get_devices(sycl::info::device_type::cpu);
    auto n_cpus=size( cpu_devices );
    std::cout << "\t\t There are "<< n_cpus << " CPUs\n"<< std::endl;
    if(n_cpus>0)
    {
        for (int i_cpu=0;i_cpu<n_cpus; i_cpu++) 
        {
            std::cout << "\t\t\t Device: " << cpu_devices[i_cpu].get_info<sycl::info::device::name >()<< "\n" << std::endl;
        }
    }


    std::cout << "\tChecking for GPUs\n" << std::endl;
    auto gpu_devices= sycl::device::get_devices(sycl::info::device_type::gpu);
    auto n_gpus=size( gpu_devices );
    std::cout << "\t\t There are "<< n_gpus << " GPUs\n"<< std::endl;
    if(n_gpus>0)
    {
        for (int i_gpu=0;i_gpu<n_gpus; i_gpu++) 
        {
            std::cout << "\t\t\t Device: " << gpu_devices[i_gpu].get_info<sycl::info::device::name >()<< "\n" << std::endl;
        }
    }

    if(n_cpus>0)
    {
        std::cout << "Checking properties of a queue CPU device\n" << std::endl;
        queue q{cpu_devices[0],q_prof};
        ShowDevice(q);
    }

    if(n_gpus>0)
    {
        std::cout << "Checking properties of a GPU device\n" << std::endl;
        queue q{gpu_devices[0],q_prof};
        ShowDevice(q);
    }
}
