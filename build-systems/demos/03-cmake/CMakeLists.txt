# Makes cmake act like its version 3.18 (this is a comment)
cmake_minimum_required(VERSION 3.18)

project(project_hello VERSION 0.1 DESCRIPTION "Hello project" LANGUAGES C) 

set(BUILDER_NAME "Default" CACHE STRING "Variable description")

configure_file (
    "${PROJECT_SOURCE_DIR}/include/config.h.in"
    "${PROJECT_BINARY_DIR}/include/config.h")

add_library(module STATIC module.c)
target_include_directories(module PUBLIC include)

add_executable(hello hello.c)
target_include_directories(hello PUBLIC include ${PROJECT_BINARY_DIR}/include)

target_link_libraries(hello PUBLIC module)

install(TARGETS hello module 
  RUNTIME DESTINATION "bin" 
  ARCHIVE DESTINATION "lib"
  LIBRARY DESTINATION "LIB")
