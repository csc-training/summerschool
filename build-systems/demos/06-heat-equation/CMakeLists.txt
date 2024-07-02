cmake_minimum_required(VERSION 3.18)

project(heat VERSION 1.0 DESCRIPTION "Heat equation w/ cmake" LANGUAGES CXX C)
find_package(OpenMP)
find_package(PNG REQUIRED)

add_executable(heat_omp core.cpp heat.cpp main.cpp 
  setup.cpp utilities.cpp io.cpp "../../../hybrid/heat-hybrid/common/pngwriter.c")
target_link_libraries(heat_omp PRIVATE png)
target_include_directories(heat_omp PUBLIC "../../../hybrid/heat-hybrid/common")

if(OpenMP_CXX_FOUND)
  target_link_libraries(heat_omp PUBLIC OpenMP::OpenMP_CXX "${OpenMP_CXX_FLAGS}")
endif()


install(TARGETS heat_omp RUNTIME DESTINATION "heat" )
install(FILES "bottle.dat" DESTINATION "heat")
