#!/bin/bash
#
# This script compiles the QuickSR project into a shared library without CMake.
# It manually calls the C++ and HIP compilers with the correct, explicit flags,
# excludes the /tests directory, and compiles in parallel using all CPU cores.
# Stack protection is disabled to match the custom toolchain.
#

# --- Script Configuration ---

# Exit immediately if a command exits with a non-zero status.
set -e

# Define color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting QuickSR parallel manual build...${NC}"

# --- Build Environment Variables ---

# Project root directory (assumes the script is in the project root)
PROJECT_ROOT=$(pwd)

# Directory for object files and the final library
BUILD_DIR="${PROJECT_ROOT}/build_manual"

# Paths to your custom compilers and libraries
XPACK_GCC_ROOT="$HOME/xpack-gcc-14.2.0-2"
ROCM_PATH="/opt/rocm"

# Compiler executables
CXX_COMPILER="${XPACK_GCC_ROOT}/bin/g++ -D__HIP_PLATFORM_AMD__"
HIP_COMPILER="${ROCM_PATH}/bin/hipcc -D__HIP_PLATFORM_AMD__"

# Python paths (using python3-config for robustness)
PYTHON_INCLUDE_DIRS=$(python3-config --cflags)
PYTHON_LINK_LIBS=$(python3-config --ldflags --embed)

# --- Compiler and Linker Flags ---

# Flags common to both C++ and HIP compilation
COMMON_FLAGS="-I${PROJECT_ROOT}/include/quicksr \
              -I${PROJECT_ROOT}/extern/pybind11/include \
              -I${ROCM_PATH}/include \
              ${PYTHON_INCLUDE_DIRS} \
              -fPIC \
              -O3 \
              -g \
              -std=c++23 \
              -march=native \
              -fopenmp \
              -Wno-format-security \
              -fno-stack-protector" # <-- THIS FLAG DISABLES STACK PROTECTION

# Flags specifically for the C++ compiler (g++)
CXX_FLAGS="${COMMON_FLAGS}"

# Flags specifically for the HIP compiler (hipcc)
HIP_FLAGS="${COMMON_FLAGS} --gcc-toolchain=${XPACK_GCC_ROOT}"

# --- Build Process ---

# 1. Clean and create the build directory
echo -e "\n${YELLOW}1. Cleaning and creating build directory: ${BUILD_DIR}${NC}"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

# 2. Find all source files, excluding the 'tests' directory
echo -e "\n${YELLOW}2. Finding source files (excluding tests)...${NC}"
CXX_SOURCES=$(find "${PROJECT_ROOT}/src" -name "*.cc")
HIP_SOURCES=$(find "${PROJECT_ROOT}/src" -name "*.hip")

# 3. Compile all C++ and HIP files in parallel
echo -e "\n${YELLOW}3. Compiling all source files in parallel...${NC}"

# Get the number of CPU cores
NUM_CORES=$(nproc)
echo "Using ${CYAN}${NUM_CORES}${NC} cores for compilation."

# Define a function to be used by xargs for compilation
export PROJECT_ROOT BUILD_DIR CXX_COMPILER CXX_FLAGS HIP_COMPILER HIP_FLAGS
compile_file() {
    src_file="$1"
    
    # Determine which compiler and flags to use based on file extension
    if [[ "${src_file}" == *.cc ]]; then
        compiler="${CXX_COMPILER}"
        flags="${CXX_FLAGS}"
        lang_tag="[CXX]"
    elif [[ "${src_file}" == *.hip ]]; then
        compiler="${HIP_COMPILER}"
        flags="${HIP_FLAGS}"
        lang_tag="[HIP]"
    else
        return
    fi
    
    # Construct the object file path
    relative_path=$(realpath --relative-to="${PROJECT_ROOT}" "${src_file}")
    obj_file="${BUILD_DIR}/${relative_path}.o"

    # Create the subdirectory in the build directory
    mkdir -p "$(dirname "${obj_file}")"
    
    # Compile the file
    echo "  ${lang_tag} Compiling ${relative_path}"
    ${compiler} ${flags} -c "${src_file}" -o "${obj_file}"
}
export -f compile_file

# Run compilation in parallel using xargs
printf "%s\n" ${CXX_SOURCES} ${HIP_SOURCES} | xargs -P "${NUM_CORES}" -I {} bash -c 'compile_file "{}"'

# 4. Link all object files into a shared library
echo -e "\n${YELLOW}4. Linking the shared library libquicksr.so...${NC}"
OBJECT_FILES=$(find "${BUILD_DIR}" -name "*.o")
OUTPUT_LIBRARY="${BUILD_DIR}/libquicksr.so"

# Link without the -lssp flag, as the feature is now disabled.
${HIP_COMPILER} -shared ${COMMON_FLAGS} -o "${OUTPUT_LIBRARY}" ${OBJECT_FILES} ${PYTHON_LINK_LIBS}

echo -e "\n${GREEN}✅ Build successful!${NC}"
echo -e "Shared library created at: ${GREEN}${OUTPUT_LIBRARY}${NC}\n"
