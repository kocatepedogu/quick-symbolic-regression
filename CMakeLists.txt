cmake_minimum_required(VERSION 3.20)
project(QuickSR LANGUAGES CXX)

# Set the HIP compiler
set(CMAKE_CXX_COMPILER hipcc)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find HIP package
find_package(HIP REQUIRED)

# Define source directories
set(SOURCE_DIRS 
    ${CMAKE_CURRENT_SOURCE_DIR}/compiler
    ${CMAKE_CURRENT_SOURCE_DIR}/dataset
    ${CMAKE_CURRENT_SOURCE_DIR}/expressions
    ${CMAKE_CURRENT_SOURCE_DIR}/vm
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/common
    ${CMAKE_CURRENT_SOURCE_DIR}/inter-individual
    ${CMAKE_CURRENT_SOURCE_DIR}/inter-individual/program
    ${CMAKE_CURRENT_SOURCE_DIR}/inter-individual/vm
    ${CMAKE_CURRENT_SOURCE_DIR}/intra-individual
    ${CMAKE_CURRENT_SOURCE_DIR}/intra-individual/program
    ${CMAKE_CURRENT_SOURCE_DIR}/intra-individual/vm
)

# Collect all source files
set(SOURCES)
foreach(DIR ${SOURCE_DIRS})
    file(GLOB DIR_SOURCES 
        "${DIR}/*.cc"
    )
    list(APPEND SOURCES ${DIR_SOURCES})
endforeach()

# Create shared library
add_library(quicksr SHARED ${SOURCES})

# Include directories
target_include_directories(quicksr PUBLIC
    ${SOURCE_DIRS}
    ${HIP_INCLUDE_DIRS}
)

# Link HIP libraries
target_link_libraries(quicksr
    ${HIP_LIBRARIES}
    hip::host
    hip::device
    -lomp
)

# Compiler options
target_compile_options(quicksr PRIVATE
    -fPIC 
    -g
    -O3
    -march=native
    -fopenmp
    -Wno-format-security
)

# Optional: Install library
install(TARGETS quicksr
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

add_executable(testintra ${CMAKE_CURRENT_SOURCE_DIR}/tests/testintra.cc)
add_executable(testinter ${CMAKE_CURRENT_SOURCE_DIR}/tests/testinter.cc)

target_link_libraries(testintra quicksr)
target_link_libraries(testinter quicksr)
