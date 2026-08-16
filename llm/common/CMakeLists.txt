find_package(Threads REQUIRED)

set(TARGET lite_llm_common)

file(GLOB LITE_LLM_COMMON_CPP_SOURCES "*.cpp")

add_library(${TARGET} ${LITE_LLM_COMMON_CPP_SOURCES})

set_target_properties(${TARGET} PROPERTIES VERSION ${LITE_LLM_VERSION_BASE} SOVERSION ${LITE_LLM_VERSION_MAJOR} MACHO_CURRENT_VERSION 0)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${TARGET} PUBLIC cxx_std_17)

if(BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_link_libraries(${TARGET}
    PUBLIC Threads::Threads glog::glog
)