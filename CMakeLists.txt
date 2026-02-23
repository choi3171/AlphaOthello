cmake_minimum_required(VERSION 3.14)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
project(cpp_selfplay LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(
  cpp_selfplay
  main.cpp
  search.cpp
  quoridor.cpp
  onnx_infer.cpp
)

target_compile_options(cpp_selfplay PRIVATE -O3 -march=native -ffast-math -flto)

if(NOT DEFINED ONNXRUNTIME_DIR)
  message(FATAL_ERROR "Set ONNXRUNTIME_DIR to your onnxruntime installation root")
endif()

target_include_directories(cpp_selfplay PRIVATE ${ONNXRUNTIME_DIR}/include)
target_include_directories(cpp_selfplay PRIVATE ${CUDAToolkit_DIR})

find_library(
  ONNXRUNTIME_LIB
  NAMES onnxruntime
  PATHS ${ONNXRUNTIME_DIR}/lib
  NO_DEFAULT_PATH
  REQUIRED
)

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

target_link_libraries(cpp_selfplay PRIVATE ${ONNXRUNTIME_LIB} Threads::Threads)