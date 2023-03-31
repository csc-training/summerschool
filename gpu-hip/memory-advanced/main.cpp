#include"example_struct.hpp"
#include<iostream>
#include<vector>

__global__ void simple_kernel(mem_wrapper<int>::dev_side data)
{
  if (threadIdx.x < data.num_elems)
    data.dev_ptr[threadIdx.x] = threadIdx.x;
}


__global__ void complex_kernel(example::dev_side data)
{
  if (threadIdx.x < data.content.intcont.num_elems) //struct.tuple.field_in_tuple.mem_wrapper
    printf("Hello from thread %d, numelems is %d, my int value is %d \n", threadIdx.x,static_cast<int>(data.content.intcont.num_elems) ,data.content.intcont.dev_ptr[threadIdx.x]);

  if (threadIdx.x < data.content.floatcont.num_elems)
    printf("Hello from thread %d, numelems is %d, my float value is %f \n",threadIdx.x,static_cast<int>(data.content.floatcont.num_elems) ,data.content.floatcont.dev_ptr[threadIdx.x]);
}

int main(){

  //1 layer wrapper example: easier (this is just a "RAII" wrapping of the data, no more complexity involved.)
  //it has to be managed manually
  mem_wrapper<int> simple_wrap(10);
  //simple_wrap.memset_dev();
  simple_kernel<<<1,32>>>(simple_wrap.get_device_representation());
  simple_wrap.cpy_to_host();
  HIP_ERRCHK(hipDeviceSynchronize());
  std::cout<< "Initialized from device: ";
  for(int i=0; i<simple_wrap.num_elems; ++i)
  {
    std::cout<<"data["<<i<<"]="<< simple_wrap.host_ptr[i] << " ";
  }
  std::cout<<std::endl;


  //2 layer wrapping example: we can organize data in more complex structures
  //which makes the functions more readable. it is however more complex to setup and manage
  
  //create the data struct of given size. look the ctor for the initialization
  example data_struct(5);
  //use it
  complex_kernel<<<1,32>>>(data_struct.get_device_representation());
  


  //create some date somewhere on the host
  std::vector<int> a = {1,2,3,4,5};
  std::vector<float> b = {5.5,4.4,3.3,2.2,1.1};
  //make a tuple with the same format of the data container
  std::tuple<std::vector<int>,std::vector<float>> c = std::make_tuple(a,b);
  
  //use a fold expression to copy all the data on the host side of the mem wrappers
  std::apply([&](auto&&... x){
    std::apply([&](auto&&... y){
        (y.copy_in(x.data()),...);
    }, data_struct.content);
  }, c);
  //copy to device
  data_struct.copy_2_dev();
  //use them!
  complex_kernel<<<1,32>>>(data_struct.get_device_representation());
  
}
