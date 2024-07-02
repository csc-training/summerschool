cmake_minimum_required(VERSION 3.18)

# Get MPI
find_package(MPI REQUIRED)

# Set variables
set(lib ${PROJECT_NAME})
set(objs objs)
set(binary ${PROJECT_NAME}-bin)

# Build a static library, object library and an executable
add_library(${lib} STATIC)
add_library(${objs} OBJECT "")
add_executable(${binary})

# List the sources of the object library
target_sources(
    ${objs}
    PRIVATE
    core.cpp
    field.cpp
    io.cpp
    lib.cpp
    parallel.cpp
    utilities.cpp
    "${common_dir}/pngwriter.c"
    )

# The executable only has main as a source, it links with the static library
target_sources(
    ${binary}
    PRIVATE
    main.cpp
    )

# Add any MPI options and definitions
target_compile_options(${objs}
    PRIVATE
    ${MPI_C_COMPILE_OPTIONS}
    )

target_compile_definitions(${objs}
    PRIVATE
    ${MPI_C_COMPILE_DEFINITIONS}
    )

# Add some include directories with the SYSTEM flag.
# This means our own compile options don't trigger warnings/errors
# from those header files
target_include_directories(${objs}
    SYSTEM
    PRIVATE
    ${MPI_C_INCLUDE_DIRS}
    ${dependencies}
    )

# Our own header files
target_include_directories(${objs}
    PRIVATE
    ${common_dir}
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

# Same for the executable
target_include_directories(${binary}
    SYSTEM
    PRIVATE
    ${MPI_C_INCLUDE_DIRS}
    ${dependencies}
    )

target_include_directories(${binary}
    PRIVATE
    ${common_dir}
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

# The static library links with MPI and the object library
target_link_options(${lib}
    PRIVATE
    ${MPI_C_LINK_FLAGS}
    )

target_link_libraries(${lib}
    PRIVATE
    ${objs}
    ${MPI_C_LIBRARIES}
    )

# The executable links with the static library
target_link_libraries(${binary} PUBLIC ${lib})

install(
    TARGETS ${lib} ${binary}
    CONFIGURATIONS Release ReleaseWithDebInfo
    ARCHIVE DESTINATION ${install_dir}/lib
    LIBRARY DESTINATION ${install_dir}/lib
    RUNTIME DESTINATION ${install_dir}/bin
    PUBLIC_HEADER DESTINATION ${install_dir}/include
    )

# Create a symlink from testdata directory to the install directory
install(DIRECTORY "${CMAKE_SOURCE_DIR}/tests/testdata" DESTINATION ${install_dir}/bin)
