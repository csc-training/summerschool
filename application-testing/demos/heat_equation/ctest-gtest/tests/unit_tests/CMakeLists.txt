cmake_minimum_required(VERSION 3.18)

set(testname ${PROJECT_NAME}_unit_tests)

add_executable(
    ${testname}
    field_tests.cpp
    io_tests.cpp
    utilities_tests.cpp
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
