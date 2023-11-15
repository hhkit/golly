cmake_minimum_required(VERSION 3.20.0)
project(Golly)

include(FetchContent)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# LLVM setup
find_package(LLVM REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(GOLLY_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(GOLLY_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

list(INSERT CMAKE_MODULE_PATH 0
  "${GOLLY_SOURCE_DIR}/cmake"
  "${LLVM_COMMON_CMAKE_UTILS}/Modules"
  )
# other libraries
find_package(ISL)

FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG a0b8a92e3d1532361c2f7feb63babc5c18d00ef2 # 10.0
  )
FetchContent_Declare(ryml
    GIT_REPOSITORY https://github.com/biojppm/rapidyaml.git
    GIT_TAG ${RYML_BRANCH_NAME}
    GIT_SHALLOW FALSE  # ensure submodules are checked out
  )
FetchContent_MakeAvailable(fmt ryml)

include_directories(${LLVM_INCLUDE_DIRS} ${ISL_INCLUDE_DIRS} include)
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})


# Now build our tools
add_subdirectory(lib)

add_subdirectory(barrier-repair)
add_subdirectory(tools)

# # Find the libraries that correspond to the LLVM components
# # that we wish to use
# llvm_map_components_to_libnames(llvm_libs support core irreader)

# # Link against LLVM libraries
# target_link_libraries(golly ${llvm_libs})