cmake_minimum_required(VERSION 3.12) 
set (TOOL_NAME golly-repair)
project(${TOOL_NAME} LANGUAGES C CXX)

find_package(LLVM REQUIRED)
find_package(Clang REQUIRED)


include_directories(include)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})

add_definitions(${LLVM_DEFINITIONS})
add_definitions(${CLANG_DEFINITIONS})

add_executable(${TOOL_NAME}
    src/main.cpp 
    src/StmtCollector.cpp
    src/SourceLocation.cpp
    )
target_link_libraries(${TOOL_NAME} clangBasic clangTooling clangTransformer clangTidyUtils)
target_compile_features(${TOOL_NAME} PRIVATE cxx_std_20)


if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
target_compile_options(${TOOL_NAME} PRIVATE "-Wno-deprecated-enum-enum-conversion")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
target_compile_options(${TOOL_NAME} PRIVATE "-Wno-deprecated-enum-enum-conversion")
endif()