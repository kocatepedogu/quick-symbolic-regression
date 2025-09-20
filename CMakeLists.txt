cmake_minimum_required(VERSION 3.20)
project(QuickSR LANGUAGES CXX)

# Options
set(ENABLE_SANITIZERS OFF)

# Find HIP package
find_package(HIP REQUIRED)

# Find Python3 Development Package
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Set the HIP language
enable_language(HIP)

# Set the HIP compiler
set(CMAKE_CXX_COMPILER hipcc)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configuration for Address Sanitizer and Undefined Behavior Sanitizer
if(ENABLE_SANITIZERS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        # Compiler Flags
        set(SANITIZER_COMPILER_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer)
        
        # Linker flags
        set(SANITIZER_LINKER_FLAGS asan ubsan)
    endif()
endif()

# Define source directories
set(SOURCE_DIRS 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/compiler
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dataset
    ${CMAKE_CURRENT_SOURCE_DIR}/src/expressions
    ${CMAKE_CURRENT_SOURCE_DIR}/src/genetic
    ${CMAKE_CURRENT_SOURCE_DIR}/src/genetic/common
    ${CMAKE_CURRENT_SOURCE_DIR}/src/genetic/initialization
    ${CMAKE_CURRENT_SOURCE_DIR}/src/genetic/initialization/initializer
    ${CMAKE_CURRENT_SOURCE_DIR}/src/genetic/recombination
    ${CMAKE_CURRENT_SOURCE_DIR}/src/genetic/recombination/recombiner
    ${CMAKE_CURRENT_SOURCE_DIR}/src/genetic/mutation
    ${CMAKE_CURRENT_SOURCE_DIR}/src/genetic/mutation/mutator
    ${CMAKE_CURRENT_SOURCE_DIR}/src/genetic/selection
    ${CMAKE_CURRENT_SOURCE_DIR}/src/genetic/selection/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/src/util
    ${CMAKE_CURRENT_SOURCE_DIR}/src/util/arrays
    ${CMAKE_CURRENT_SOURCE_DIR}/src/python
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runners
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runners/cpu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runners/hybrid
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runners/inter-individual
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runners/inter-individual/cache
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runners/inter-individual/program
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runners/intra-individual
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runners/intra-individual/program
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runners/intra-individual/cache
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/common
)

# Collect all source files
set(SOURCES)
foreach(DIR ${SOURCE_DIRS})
    file(GLOB DIR_SOURCES 
        "${DIR}/*.hip"
        "${DIR}/*.cc"
    )
    list(APPEND SOURCES ${DIR_SOURCES})
endforeach()

# Create shared library
add_library(quicksr SHARED ${SOURCES})

# Include directories
target_include_directories(quicksr PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/quicksr
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11/include
    ${HIP_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

# Link HIP libraries
target_link_libraries(quicksr
    ${Python3_LIBRARIES}
    ${HIP_LIBRARIES}
    hip::host
    hip::device
    -fopenmp
    ${SANITIZER_LINKER_FLAGS}
)

# Compiler options
target_compile_options(quicksr PRIVATE
    -fPIC 
    -g
    -O3
    -march=native
    -fopenmp
    -Wno-format-security
    ${SANITIZER_COMPILER_FLAGS}
)

# Optional: Install library
install(TARGETS quicksr
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

add_executable(testintra ${CMAKE_CURRENT_SOURCE_DIR}/tests/testintra.cc)
add_executable(testinter ${CMAKE_CURRENT_SOURCE_DIR}/tests/testinter.cc)
add_executable(testcpu ${CMAKE_CURRENT_SOURCE_DIR}/tests/testcpu.cc)
add_executable(testhybrid ${CMAKE_CURRENT_SOURCE_DIR}/tests/testhybrid.cc)
add_executable(testexprgen ${CMAKE_CURRENT_SOURCE_DIR}/tests/testexprgen.cc)
add_executable(testmutation ${CMAKE_CURRENT_SOURCE_DIR}/tests/testmutation.cc)
add_executable(testcrossover ${CMAKE_CURRENT_SOURCE_DIR}/tests/testcrossover.cc)
add_executable(testexprpick ${CMAKE_CURRENT_SOURCE_DIR}/tests/testexprpick.cc)

target_compile_options(testintra PRIVATE -O3 -march=native -fopenmp ${SANITIZER_COMPILER_FLAGS})
target_compile_options(testinter PRIVATE -O3 -march=native -fopenmp ${SANITIZER_COMPILER_FLAGS})
target_compile_options(testcpu PRIVATE -O3 -march=native -fopenmp ${SANITIZER_COMPILER_FLAGS})
target_compile_options(testhybrid PRIVATE -O3 -march=native -fopenmp ${SANITIZER_COMPILER_FLAGS})
target_compile_options(testexprgen PRIVATE -O3 -march=native -fopenmp ${SANITIZER_COMPILER_FLAGS})
target_compile_options(testmutation PRIVATE -O3 -march=native -fopenmp ${SANITIZER_COMPILER_FLAGS})
target_compile_options(testcrossover PRIVATE -O3 -march=native -fopenmp ${SANITIZER_COMPILER_FLAGS})
target_compile_options(testexprpick PRIVATE -O3 -march=native -fopenmp ${SANITIZER_COMPILER_FLAGS})

target_link_libraries(testintra ${SANITIZER_LINKER_FLAGS} quicksr)
target_link_libraries(testinter ${SANITIZER_LINKER_FLAGS} quicksr)
target_link_libraries(testcpu ${SANITIZER_LINKER_FLAGS} quicksr)
target_link_libraries(testhybrid ${SANITIZER_LINKER_FLAGS} quicksr)
target_link_libraries(testexprgen ${SANITIZER_LINKER_FLAGS} quicksr)
target_link_libraries(testmutation ${SANITIZER_LINKER_FLAGS} quicksr)
target_link_libraries(testcrossover ${SANITIZER_LINKER_FLAGS} quicksr)
target_link_libraries(testexprpick ${SANITIZER_LINKER_FLAGS} quicksr)
