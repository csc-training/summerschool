cmake_minimum_required(VERSION 3.18)

set(googletest googletest-1.14.0)
set(googletest_include ${CMAKE_CURRENT_SOURCE_DIR}/${googletest}/googletest/include)

set(unittests unit_tests)

add_subdirectory(${googletest})
add_subdirectory(${unittests})
