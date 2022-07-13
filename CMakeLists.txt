cmake_minimum_required(VERSION 3.18)

project(canny VERSION 1.0.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

# use c++ 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# use static libraries
set(BUILD_SHARED_LIBS OFF)

set(CPP_SOURCE canny.cpp)
set(HEADERS canny.h)

# add directory for find package to search
# list(APPEND CMAKE_PREFIX_PATH "$HOME/libtorch")

# find opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# add local files
add_library(
  canny.hpp
  canny.cpp
)


add_executable(canny main.cpp)

target_link_libraries(canny "${OpenCV_LIBS}" canny.hpp)

set_property(TARGET canny PROPERTY CXX_STANDARD 17)
