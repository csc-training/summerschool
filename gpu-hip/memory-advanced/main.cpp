#include<example_struct.hpp>

__global__ void simple_kernel(mem_wrapper<int>::dev_side data)
{
  if (threadIdx.x < data.num_elem)
    data.dev_ptr[threadIdx.x] = threadIdx.x;
}

__global__ void complex_kernel(example::dev_side data)
{
  if (threadIdx.x < data.intcont.num_elems)
    printf("Hello from thread %d, my int value is %d and my float value is %f \n", threadIdx.x, data.content.intcont.dev_ptr[threadIdx.x],data.content.floatcont.dev_ptr[threadIdx.x]);
}

int main(){
  mem_wrapper<int> simple_wrap(10);
  simple_wrap.memset_dev();
  simple_kernel<<<1,32>>>(simple_wrap.get_devside());
  simple_wrap.cpy_to_host();
  hipDeviceSynchronize();
  std::cout<< "Initialized from device: ";
  for(int i=0; i<simple_wrap.num_elems; ++i)
  {
    std::cout<<"data["<<i<<"]="<< simple_wrap.host_ptr[i] << " "
  }


  example data_struct(5);
  complex_kernel<<<1,32>>>(data_struct.get_devside());

}