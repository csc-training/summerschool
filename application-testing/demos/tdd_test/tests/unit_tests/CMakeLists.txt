cmake_minimum_required(VERSION 3.18)

set(testname ${PROJECT_NAME}_tdd_tests)

add_executable(
    ${testname}
    field_tests.cpp
)

target_include_directories(${testname} SYSTEM PRIVATE
    "${googletest_include}"
    )

target_link_libraries(
  ${testname}
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(${testname})
