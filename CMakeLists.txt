cmake_minimum_required(VERSION 3.18)
project(neural_net LANGUAGES C CXX CUDA)

# ==============================================================================
# Global Configuration
# ==============================================================================

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type" FORCE)
endif()

# Custom build types
set(CMAKE_CONFIGURATION_TYPES
    "Debug;Release;Perf;Profile"
    CACHE STRING "" FORCE)

# ==============================================================================
# Compiler Detection & Flags
# ==============================================================================

# Prefer clang if available
if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
  message(STATUS "Using Clang compiler")
elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  message(STATUS "Using GCC compiler")
endif()

# Base C flags (all builds)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -D_POSIX_C_SOURCE=199309L")

# Release flags
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -mavx512f -flto -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mavx512f -flto -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# Debug flags
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG -march=native -mavx512f")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -march=native -mavx512f")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")

# Perf profiling flags (custom build type)
set(CMAKE_C_FLAGS_PERF
    "-O3 -march=native -mavx512f -flto -g -fno-omit-frame-pointer -fno-inline -DNDEBUG -DPROFILING=1"
)
set(CMAKE_CXX_FLAGS_PERF
    "-O3 -march=native -mavx512f -flto -g -fno-omit-frame-pointer -fno-inline -DNDEBUG -DPROFILING=1"
)
set(CMAKE_EXE_LINKER_FLAGS_PERF "${CMAKE_EXE_LINKER_FLAGS_RELEASE}")

# Instruction-level profiling flags (custom build type)
set(CMAKE_C_FLAGS_PROFILE
    "-O3 -march=native -DPROFILING=1 -fprofile-instr-generate -fcoverage-mapping"
)
set(CMAKE_CXX_FLAGS_PROFILE
    "-O3 -march=native -DPROFILING=1 -fprofile-instr-generate -fcoverage-mapping"
)
set(CMAKE_EXE_LINKER_FLAGS_PROFILE "-fprofile-instr-generate -flto")

# NVCC flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -Wall")

# ==============================================================================
# CUDA Configuration
# ==============================================================================

find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures (adjust for your GPU) 89 = Ada Lovelace (RTX 40
# series), 86 = Ampere (RTX 30 series)
set(CMAKE_CUDA_ARCHITECTURES
    89
    CACHE STRING "CUDA architectures")

# ==============================================================================
# Source Discovery
# ==============================================================================

file(GLOB_RECURSE C_SOURCES "${CMAKE_SOURCE_DIR}/src/*.c")
file(GLOB_RECURSE CUDA_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cu")
file(GLOB_RECURSE HEADERS "${CMAKE_SOURCE_DIR}/src/*.h")
file(GLOB TEST_SOURCES "${CMAKE_SOURCE_DIR}/tests/*.c")

# Separate main.c from library sources
list(FILTER C_SOURCES EXCLUDE REGEX ".*main\\.c$")
set(MAIN_SOURCE "${CMAKE_SOURCE_DIR}/src/main.c")

# ==============================================================================
# Core Library (shared by all targets)
# ==============================================================================

add_library(nn_core STATIC ${C_SOURCES} ${CUDA_SOURCES})

target_include_directories(nn_core PUBLIC ${CMAKE_SOURCE_DIR}/src
                                          ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(nn_core PUBLIC CUDA::cudart CUDA::cublas m)

# PIC for shared library compatibility (Python bindings)
set_target_properties(nn_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ==============================================================================
# Main Executable
# ==============================================================================

add_executable(neural_net ${MAIN_SOURCE})
target_link_libraries(neural_net PRIVATE nn_core)

# Force clang++ as linker when using clang with LTO
if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
  set_target_properties(neural_net PROPERTIES LINKER_LANGUAGE CXX)
endif()

# ==============================================================================
# Test Runner
# ==============================================================================

add_executable(test_runner ${TEST_SOURCES})
target_include_directories(test_runner PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_runner PRIVATE nn_core)

# Force clang++ as linker when using clang with LTO
if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
  set_target_properties(test_runner PROPERTIES LINKER_LANGUAGE CXX)
endif()

enable_testing()
add_test(NAME unit_tests COMMAND test_runner)

# ==============================================================================
# Python Bindings (pybind11)
# ==============================================================================

option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

if(BUILD_PYTHON_BINDINGS)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0)
  FetchContent_MakeAvailable(pybind11)

  # Binding sources - add your files here as you implement them
  set(BINDING_SOURCES python/bindings/module.cpp)

  pybind11_add_module(_nn_core ${BINDING_SOURCES})

  # Output extension directly to Python package for unified build
  set_target_properties(_nn_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/nn_c)

  # Link to nn_core library and explicitly inherit include directories
  target_link_libraries(_nn_core PRIVATE nn_core)
  target_include_directories(
    _nn_core PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/core
                     ${Python_INCLUDE_DIRS})

  # Install the extension module inside the nn_c package for proper stub discovery
  install(TARGETS _nn_core LIBRARY DESTINATION nn_c)
endif()

# ==============================================================================
# Convenience Targets
# ==============================================================================

# Run target
add_custom_target(
  run
  COMMAND neural_net
  DEPENDS neural_net
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Running neural_net")

# Test target
add_custom_target(
  check
  COMMAND test_runner
  DEPENDS test_runner
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Running tests")

# Memcheck target
add_custom_target(
  memcheck
  COMMAND valgrind --error-exitcode=1 --leak-check=full --show-leak-kinds=all
          --track-origins=yes $<TARGET_FILE:neural_net>
  DEPENDS neural_net
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Running valgrind memcheck")

# Format target for all source files (C/C++ and Python)
add_custom_target(
  format
  COMMAND find ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests -name "*.c" -o
          -name "*.cu" -o -name "*.h" | xargs clang-format -i
  COMMAND uv run ruff format ${CMAKE_SOURCE_DIR}/python
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Formatting C/C++ and Python source files")

# Perf recording (requires perf build)
add_custom_target(
  perf-record
  COMMAND sudo perf record -g -F 4000 --call-graph dwarf
          $<TARGET_FILE:neural_net>
  DEPENDS neural_net
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Recording perf profile")

add_custom_target(
  perf-report
  COMMAND sudo perf report --stdio --no-children -n | head -100
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Generating perf report")

# Python test target (only if bindings are enabled)
if(BUILD_PYTHON_BINDINGS)
  add_custom_target(
    pytest
    COMMAND
      ${CMAKE_COMMAND} -E echo
      "Note: Run 'uv pip install -e .' first to install the package in development mode"
    COMMAND ${Python_EXECUTABLE} -m pytest python/tests/ -v ||
            ${Python_EXECUTABLE} python/run_tests.py
    DEPENDS _nn_core
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running Python tests")
endif()

# ==============================================================================
# Installation
# ==============================================================================

install(TARGETS neural_net RUNTIME DESTINATION bin)
install(TARGETS nn_core ARCHIVE DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/nn_c)

# ==============================================================================
# Summary
# ==============================================================================

message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "")
message(STATUS "Build targets:")
message(STATUS "  neural_net    - Main executable")
message(STATUS "  test_runner   - Test executable")
message(STATUS "  nn_core       - Static library")
if(BUILD_PYTHON_BINDINGS)
  message(STATUS "  _nn_core      - Python extension module")
endif()
message(STATUS "")
message(STATUS "Convenience targets:")
message(STATUS "  run           - Build and run neural_net")
message(STATUS "  check         - Build and run tests")
message(STATUS "  memcheck      - Run with valgrind")
message(STATUS "  format        - Format C/C++ (clang-format) and Python (ruff)")
message(STATUS "  perf-record   - Record perf profile")
message(STATUS "  perf-report   - Show perf report")
if(BUILD_PYTHON_BINDINGS)
  message(STATUS "  pytest        - Run Python bindings tests")
endif()
message(STATUS "")
