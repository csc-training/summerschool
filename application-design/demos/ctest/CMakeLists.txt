cmake_minimum_required(VERSION 3.20)

project(test)
enable_testing()


add_executable(out main.cpp)

add_test(
	NAME test_1
	COMMAND out
	)

set_tests_properties(test_1 PROPERTIES PASS_REGULAR_EXPRESSION "hello world")
