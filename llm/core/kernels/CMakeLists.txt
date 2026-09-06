add_subdirectory(cpu)
# add_subdirectory(cuda)

set(TARGET lite_llm_kernels)

set(ABSL_PROPAGATE_CXX_STD ON CACHE INTERNAL "")

file(GLOB LITE_LLM_KERNEL_CPP_SOURCES CONFIGURA_DEPENDS "*.cpp")
add_library(${TARGET} ${LITE_LLM_KERNEL_CPP_SOURCES})


# target_compile_definitions(${TARGET} PRIVATE ENABLE_CUDA)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ${absl_SOURCE_DIR})
target_compile_features(${TARGET} PRIVATE  cxx_std_17)

set_target_properties(${TARGET} PROPERTIES VERSION ${LITE_LLM_VERSION_BASE} SOVERSION ${LITE_LLM_VERSION_MAJOR} MACHO_CURRENT_VERSION 0)

target_link_libraries(${TARGET} PRIVATE absl::strings)
target_link_libraries(${TARGET} PUBLIC lite_llm_common)
target_link_libraries(${TARGET} PUBLIC lite_llm_framework)

if(BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()