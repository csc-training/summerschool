cmake_minimum_required(VERSION 3.18)

find_package(MPI REQUIRED)

set(testname ${PROJECT_NAME}_mpi_tests)

add_executable(
    ${testname}
    main.cpp
    core_tests.cpp
    full_run_tests.cpp
    utilities_tests.cpp
)

target_include_directories(${testname} PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    ${common_dir}
    )

target_include_directories(${testname} SYSTEM PRIVATE
    "${googletest_include}"
    "${CMAKE_SOURCE_DIR}/include"
    ${MPI_C_INCLUDE_DIRS}
    )

target_link_libraries(
  ${testname}
  GTest::gtest_main
  ${PROJECT_NAME}
)

# Add symbolic link from tests/testdata in source directory to build directory
ADD_CUSTOM_TARGET(mpi_tests_symlink ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    "${data_src_dir}"
    "${CMAKE_CURRENT_BINARY_DIR}/${data_dirname}"
    )

include(GoogleTest)

# Run the tests with mpiexec
# Need to add before gtest_add_tests to have effect

# !!!!
# N.B. using gtest_discover_tests instead of gtest_add_tests
# adds the tests X times,
# where X is the number of processes used for mpi
# In other words "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4"
# runs all the discovered tests four times
# !!!!
set_property(TARGET ${testname}
    PROPERTY CROSSCOMPILING_EMULATOR
    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
    )

gtest_add_tests(TARGET ${testname})
