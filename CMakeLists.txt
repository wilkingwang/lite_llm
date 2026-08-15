cmake_minimum_required(VERSION 3.16)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the CUDA compiler path or name
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.4/bin/nvcc")

project("lite_llm" CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

# Version
set(LITE_LLM_VERSION_MAJOR 0)
set(LITE_LLM_VERSION_MINOR 1)
set(LITE_LLM_VERSION_PATCH 0)
set(LITE_LLM_VERSION_BASE "${LITE_LLM_VERSION_MAJOR}.${LITE_LLM_VERSION_MINOR}.${LITE_LLM_VERSION_PATCH}")
set(LITE_LLM_VERSION "${LITE_LLM_VERSION_BASE}_dev")
message(STATUS "lite_llm version: ${LITE_LLM_VERSION}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
message("CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

set(LITE_LLM_PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/llm/core/framework/ic_tensor.h)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(LITE_LLM_STANDALONE ON)
else()
    set(LITE_LLM_STANDALONE OFF)
endif()

# shared vs static
option(BUILD_SHARED_LIBS "build shared libraries" ON)

# options
option(LITE_LLM_BUILD_TESTS       "lite llm: build tests"       ${LITE_LLM_STANDALONE})
option(LITE_LLM_BUILD_EXAMPLES    "lite infer: build examples"  ${LITE_LLM_STANDALONE})

# core library
add_subdirectory(llm)

if(LITE_LLM_BUILD_TESTS)
    add_subdirectory(tests)
endif()

set_target_properties(lite_llm PROPERTIES PUBLIC_HEADER "${LITE_LLM_PUBLIC_HEADERS}")