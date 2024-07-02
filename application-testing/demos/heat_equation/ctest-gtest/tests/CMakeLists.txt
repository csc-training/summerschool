cmake_minimum_required(VERSION 3.18)

set(googletest googletest-1.14.0)
set(googletest_include ${CMAKE_CURRENT_SOURCE_DIR}/${googletest}/googletest/include)

set(unittests unit_tests)
set(integrationtests integration_tests)
set(mpitests mpi_tests)

set(data_dirname testdata)
set(data_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/${data_dirname})

add_subdirectory(${googletest})
add_subdirectory(${unittests})
add_subdirectory(${integrationtests})
add_subdirectory(${mpitests})
