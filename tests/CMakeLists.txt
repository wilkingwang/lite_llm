# Build a test executable from source files.
function(lite_llm_build source)
    set(TEST_SOURCES ${source} ${ARGN})
    if(DEFINED LITE_LLM_TEST_NAME)
        set(TEST_TARGET ${LITE_LLM_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE lite_llm lite_llm_common)
endfunction()

# Register a test target with ctest
function(lite_infer_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABELWORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LITE_LLM_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LITE_LLM_TEST_LABEL)
        set(LITE_LLM_TEST_LABEL "main")
    endif()

    if(NOT DEFINED LITE_LLM_TEST_WORKING_DIRECTORY)
        set(LITE_LLM_TEST_WORKING_DIRECTORY .)
    endif()

    if (DEFINED LITE_LLM_TEST_NAME)
        set(TEST_NAME ${LITE_LLM_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LITE_LLM_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${target}>
        ${LITE_LLM_TEST_ARGS})
    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LITE_LLM_TEST_LABEL})
endfunction()

# Build and register a test in one call
function(lite_llm_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LITE_LLM_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LITE_LLM_TEST_UNPARSED_ARGUMENTS})
    if (NOT DEFINED LITE_LLM_TEST_LABEL)
        set(LITE_LLM_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LITE_LLM_TEST_WORKING_DIRECTORY)
        set(LITE_LLM_TEST_WORKING_DIRECTORY .)
    endif()

    if (DEFINED LITE_LLM_TEST_NAME)
        set(TEST_TARGET ${LITE_LLM_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE lite_llm lite_llm_common)

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LITE_LLM_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LITE_LLM_TEST_ARGS})
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LITE_LLM_TEST_LABEL})
endfunction()

# tests
lite_llm_build_and_test(test_chat.cpp)