cmake_minimum_required(VERSION 3.18)

project(something_kokkos VERSION 0.1 DESCRIPTION "Computes something with kokkos" LANGUAGES C CXX) 


# Search internet for "cmake kokkos find_package"
set(Kokkos_ROOT "/home/jkataja/opt/kokkos/4.2.01" CACHE PATH "Where to find kokkos")
set(Kokkos_DIR "${Kokkos_ROOT}" CACHE PATH "kokkos_DIR" FORCE) # Non cache variable!
# set(Kokkos_DIR "/home/jkataja/opt/kokkos/4.2.01" CACHE PATH "Kokkos" directory)
find_package(Kokkos REQUIRED) 

set(CMAKE_EXPORT_COMPILE_COMMANDS true)

add_executable(something reduce.cpp)
target_link_libraries(something Kokkos::kokkos)
