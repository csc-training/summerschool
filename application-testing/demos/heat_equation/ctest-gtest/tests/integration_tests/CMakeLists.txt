cmake_minimum_required(VERSION 3.18)

set(testname ${PROJECT_NAME}_integration_tests)

add_executable(
    ${testname}
    io_tests.cpp
)

target_include_directories(${testname} PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    ${common_dir}
    )

target_include_directories(${testname} SYSTEM PRIVATE
    "${googletest_include}"
    "${CMAKE_SOURCE_DIR}/include"
    )

target_link_libraries(
  ${testname}
  GTest::gtest_main
  ${PROJECT_NAME}
)

include(GoogleTest)
gtest_discover_tests(${testname})

# Add symbolic link from tests/testdata in source directory to build directory
ADD_CUSTOM_TARGET(integration_test_symlink ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    "${data_src_dir}"
    "${CMAKE_CURRENT_BINARY_DIR}/${data_dirname}"
    )
