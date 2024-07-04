cmake_minimum_required(VERSION 3.18)

project(something_kokkos VERSION 0.1 DESCRIPTION "Computes something with kokkos" LANGUAGES C CXX) 


# Search internet for "cmake kokkos find_package"
add_subdirectory(kokkos)

set(CMAKE_EXPORT_COMPILE_COMMANDS true)

add_executable(something reduce.cpp)
target_link_libraries(something Kokkos::kokkos)

install(TARGETS something
  RUNTIME DESTINATION "bin" 
  ARCHIVE DESTINATION "lib"
  LIBRARY DESTINATION "LIB")
