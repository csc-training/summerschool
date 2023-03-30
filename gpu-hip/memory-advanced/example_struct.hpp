#include<tuple>

#include <mem_wrapper.hpp>
#include <mem_utils.hpp>

struct example{

  std::tuple<mem_wrapper<int>, mem_wrapper<float>> content;

  example(std::size_t size_int, std::size_t size_float);
  example() = delete;
  example(int size): content{mem_wrapper<int>(size), mem_wrapper<float>(size)}
  {
    content.memset_dev(0,5);
  }
  example(const example&)=delete;
  example(example&&)=default;

  //we have two "wrapping" layers that can be used to semantically group data.
  //the complete idea behind using this approach is that one "example" struct contains all the data needed to run the kernel, while
  //we still want to semantically group the data (i.e. if there is a set of coordinates, we would like them to be a "group of data")
  //in this way it should be easier to remember what is a data related to
  //in this simple example, we can reach the pointer to the intcont data from outside from the struct returned by get_devside by going into the fields "struct.content.intcont.dev_ptr"
  //it is easy to understand that if we have different sets of coordinates we can have a struct wrapping them with meaningful names 
  /*eg:
  using coordinates = std::tuple<mem_wrapper<double>,mem_wrapper<double>,mem_wrapper<double>>;
  coordinates abs_coord;
  coordinates rel_coord;


  struct coord_dev
  {
    mem_wrapper<double>::dev_side x,
    mem_wrapper<double>::dev_side y,
    mem_wrapper<double>::dev_side z,
  }
  struct dev_side {
    coord_dev abs_coord;
    coord_dev rel_coord;
  }
  we can call the coordinates with "struct.abs_coord.x.dev_ptr" or "struct.rel_coord.x.dev_ptr" from the kernel
  */

  //struct for the content tuple
  struct content_devside
  {
    mem_wrapper<int>::dev_side intcont;
    mem_wrapper<float>::dev_side floatcont;
  };

  //global dev side of the example struct
  struct dev_side
  {
    content_devside content;
  };
  
  void copy_2_dev(){
    std::apply([&](auto&... values){values.cpy_to_dev();},content);
  }

  inline auto get_devside()
  {
    //tuple 2 struct here
    auto tup = std::apply([&](auto&... values){return std::make_tuple(values.get_device_representation()...); },content);
    return dev_side{
      make_struct< content_devside > (tup)
    };
  }

};
