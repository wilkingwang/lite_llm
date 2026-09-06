set(TARGET lite_llm)

include(cmake/CPM.cmake)

CPMAddPackage(
    NAME absl
    GITHUB_REPOSITORY abseil/abseil-cpp
    GIT_TAG 20240722.0
    OPTIONS "BUILD_TESTING Off" "ABSL_PROPAGATE_CXX_STD ON" "ABSL_ENABLE_INSTALL ON"
)

CPMAddPackage(
    NAME re2
    GITHUB_REPOSITORY google/re2
    GIT_TAG 2024-07-02
)

CPMAddPackage(
    NAME nlohmann_json
    GITHUB_REPOSITORY nlohmann/json
    VERSION 3.11.3
)

add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(models)

add_library(${TARGET} lite_infer.cpp)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${TARGET} PUBLIC cxx_std_17)

set_target_properties(${TARGET} PROPERTIES VERSION ${LITE_LLM_VERSION_BASE} SOVERSION ${LITE_LLM_VERSION_MAJOR} MACHO_CURRENT_VERSION 0)

if(BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(${TARGET} PRIVATE LITE_LLM_BUILD)
    target_compile_definitions(${TARGET} PUBLIC LITE_LLM_SHARED)
endif()