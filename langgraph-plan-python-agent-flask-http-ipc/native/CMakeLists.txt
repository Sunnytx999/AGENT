cmake_minimum_required(VERSION 3.16)
project(agent_tools LANGUAGES CXX)

add_library(agent_tools SHARED src/agent_tools.cpp)
target_include_directories(agent_tools PRIVATE include)
target_compile_features(agent_tools PRIVATE cxx_std_17)
target_compile_definitions(agent_tools PRIVATE _CRT_SECURE_NO_WARNINGS)

if(MSVC)
    target_compile_options(agent_tools PRIVATE /W4 /EHsc /utf-8)
else()
    target_compile_options(agent_tools PRIVATE -Wall -Wextra -Wpedantic)
endif()

set_target_properties(agent_tools PROPERTIES
    OUTPUT_NAME "agent_tools_http_ipc_v2"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/lib"
)
