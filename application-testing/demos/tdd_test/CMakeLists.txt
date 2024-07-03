cmake_minimum_required(VERSION 3.18)

project(${project_name} VERSION 1.0 DESCRIPTION "TDD demo" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(install_dir "${PROJECT_NAME}")

option(package_tests "Build tests" ON)
if(package_tests)
    include(CTest)
    add_subdirectory(tests)
    enable_testing()
endif()
